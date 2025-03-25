#!/usr/bin/env python3
"""
Dataset Generator for Testing the Reversal Curse in LLMs

This script generates synthetic relationship data and corresponding questions to test
the "reversal curse" phenomenon in language models.
"""

import argparse
import csv
import json
import logging
import networkx as nx
import os
import random
import yaml
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import asyncio
from typing import Dict, List, Tuple, Set, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RelationshipGraphGenerator:
    """Generates a directed acyclic graph (DAG) of character relationships."""
    
    def __init__(self, config: dict):
        self.config = config
        self.characters = []
        self.relations = []
        self.graph = {}
        self.nx_graph = nx.DiGraph()
    
    def load_characters(self) -> List[str]:
        """Load character names from the specified file."""
        char_file = self.config['characters_file']
        with open(char_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_relations(self) -> List[Tuple[str, str]]:
        """Load relation pairs (forward, backward) from the specified file."""
        rel_file = self.config['relations_file']
        relations = []
        with open(rel_file, 'r') as f:
            for line in f:
                if line.strip():
                    forward, backward = line.strip().split(',')
                    relations.append((forward.strip(), backward.strip()))
        return relations
    
    def generate_graph(self) -> Dict:
        """Generate a relationship graph based on the configuration."""
        # Load characters and relations
        all_characters = self.load_characters()
        self.relations = self.load_relations()
        
        # Sample characters
        num_chars = self.config['num_characters']
        if num_chars > len(all_characters):
            logger.warning(f"Requested {num_chars} characters but only {len(all_characters)} available.")
            num_chars = len(all_characters)
        
        # Set random seed if provided
        if 'seed' in self.config:
            random.seed(self.config['seed'])
        
        # Sample characters and assign topological order
        self.characters = random.sample(all_characters, num_chars)
        
        # Initialize graph structure
        adjacency_list = {char: [] for char in self.characters}
        
        # Generate relationships following topological order
        for i, source in enumerate(self.characters):
            # Determine number of outgoing relationships
            min_rel = self.config['min_relations']
            max_rel = min(self.config['max_relations'], num_chars - i - 1)
            
            if max_rel < min_rel:
                continue  # Not enough characters left to satisfy minimum
            
            num_relations = random.randint(min_rel, max_rel)
            
            # Select target characters (only those later in topological order)
            available_targets = self.characters[i+1:]
            if len(available_targets) < num_relations:
                num_relations = len(available_targets)
            
            targets = random.sample(available_targets, num_relations)
            
            # Assign relations
            for target in targets:
                rel_idx = random.randint(0, len(self.relations) - 1)
                forward_rel, _ = self.relations[rel_idx]
                adjacency_list[source].append({"target": target, "relation": forward_rel})
                
                # Add edge to NetworkX graph
                self.nx_graph.add_edge(source, target, relation=forward_rel)
        
        # Store the generated graph
        self.graph = {
            "characters": self.characters,
            "adjacency_list": adjacency_list
        }
        
        return self.graph
    
    def get_all_relationships(self) -> List[Dict]:
        """Extract all relationships from the graph."""
        relationships = []
        for source, targets in self.graph["adjacency_list"].items():
            for edge in targets:
                target = edge["target"]
                forward_rel = edge["relation"]
                
                # Find backward relation
                backward_rel = None
                for fwd, bwd in self.relations:
                    if fwd == forward_rel:
                        backward_rel = bwd
                        break
                
                relationships.append({
                    "character_a": source,
                    "character_b": target,
                    "forward_relation": forward_rel,
                    "backward_relation": backward_rel
                })
        
        return relationships
    
    def export_graph_visualization(self, output_dir: str):
        """Export a visualization of the graph in DOT format."""
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Relationship Graph')
            
            # Add nodes
            for char in self.characters:
                dot.node(char)
            
            # Add edges
            for source, targets in self.graph["adjacency_list"].items():
                for edge in targets:
                    target = edge["target"]
                    rel = edge["relation"]
                    dot.edge(source, target, label=rel)
            
            # Save dot file and render
            dot_file = os.path.join(output_dir, 'relationship_graph')
            dot.render(dot_file, format='png')
            logger.info(f"Graph visualization saved to {dot_file}.png")
        except ImportError:
            logger.warning("Graphviz not available. Skipping visualization export.")


class QuestionGenerator:
    """Generates natural language questions using an LLM API."""
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the OpenRouter client."""
        try:
            import os
            from dotenv import load_dotenv
            from openai import AsyncOpenAI
            
            # Load environment variables from .env file
            load_dotenv()
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            
            client = AsyncOpenAI(
                api_key=api_key,
            )
            return client
        except ImportError:
            logger.warning("Required packages not installed. Using template questions.")
            return None
    
    async def generate_questions(self, relationships: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate forward train, forward test, and backward questions for all relationships asynchronously."""
        forward_train_questions = []
        forward_test_questions = []
        backward_questions = []
        
        if self.llm_client:
            # Prepare all the async tasks
            forward_tasks = []
            backward_tasks = []
            
            for rel in relationships:
                src = rel["character_a"]
                tgt = rel["character_b"]
                fwd_rel = rel["forward_relation"]
                bwd_rel = rel["backward_relation"]
                
                # Create tasks for async execution
                forward_tasks.append(self._get_question_data(src, tgt, fwd_rel, "forward"))
                backward_tasks.append(self._get_question_data(tgt, src, bwd_rel, "backward"))
            
            # Execute all forward question tasks in parallel
            if forward_tasks:
                forward_train_questions = await asyncio.gather(*forward_tasks)
                
                # Create copies for test set - will be paraphrased later
                forward_test_questions = [q.copy() for q in forward_train_questions]
            
            # Execute all backward question tasks in parallel
            if backward_tasks:
                backward_questions = await asyncio.gather(*backward_tasks)
        else:
            # Fallback to synchronous if no client
            for rel in relationships:
                src = rel["character_a"]
                tgt = rel["character_b"]
                fwd_rel = rel["forward_relation"]
                bwd_rel = rel["backward_relation"]
                
                # Forward question
                fwd_question = f"Who is {src}'s {fwd_rel}?"
                question_data = {
                    "question": fwd_question,
                    "answer": tgt,
                    "character_a": src,
                    "character_b": tgt,
                    "relation": fwd_rel
                }
                forward_train_questions.append(question_data)
                forward_test_questions.append(question_data.copy())
                
                # Backward question
                bwd_question = f"Who is {tgt}'s {bwd_rel}?"
                backward_questions.append({
                    "question": bwd_question,
                    "answer": src,
                    "character_a": tgt,
                    "character_b": src,
                    "relation": bwd_rel
                })
        
        return forward_train_questions, forward_test_questions, backward_questions
    
    async def _get_question_data(self, char_a: str, char_b: str, relation: str, direction: str) -> Dict:
        """Generate question data asynchronously."""
        question = await self._get_llm_question(char_a, relation, direction)
        
        if direction == "forward":
            return {
                "question": question,
                "answer": char_b,
                "character_a": char_a,
                "character_b": char_b,
                "relation": relation
            }
        else:  # backward
            return {
                "question": question,
                "answer": char_b,
                "character_a": char_a,
                "character_b": char_b,
                "relation": relation
            }
    
    async def _get_llm_question(self, character: str, relation: str, direction: str) -> str:
        """Generate a natural language question using OpenRouter API asynchronously."""
        if not self.llm_client:
            return f"Who is {character}'s {relation}?"
        
        prompt = f"""Generate a natural-sounding question asking who is {character}'s {relation}.
The question MUST use the exact relationship term '{relation}' without substituting other words.

Examples:
1. Character: David, Relation: mother
   Question: "Who is David's mother?"

2. Character: Sarah, Relation: child
   Question: "Who is Sarah's child?"

3. Character: Michael, Relation: mentor
   Question: "Who is Michael's mentor?"

4. Character: Emma, Relation: spouse
   Question: "Who is Emma's spouse?"

Please generate one question for Character: {character}, Relation: {relation}
Your response should contain ONLY the question and nothing else."""
        
        try:
            model = self.config.get('model')
            
            completion = await self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            question = completion.choices[0].message.content.strip()
            # Clean any unnecessary quotes
            return self._clean_quotes(question)
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Who is {character}'s {relation}?"
    
    async def paraphrase_questions(self, questions: List[Dict]) -> List[Dict]:
        """Use LLM to paraphrase questions for variety asynchronously."""
        if not self.llm_client:
            return questions
        
        # Create all tasks
        tasks = []
        for q in questions:
            tasks.append(self._paraphrase_question(q))
        
        # Execute all tasks in parallel
        if tasks:
            return await asyncio.gather(*tasks)
        return questions
    
    async def _paraphrase_question(self, question_data: Dict) -> Dict:
        """Paraphrase a single question asynchronously."""
        original = question_data["question"]
        relation = question_data["relation"]
        character_a = question_data["character_a"]
        
        try:
            model = self.config.get('model')
            
            prompt = f"""Paraphrase this question while preserving its exact meaning: '{original}'

The paraphrased question MUST:
1. Keep the exact same relationship term '{relation}'
2. Keep the character name '{character_a}'
3. Still clearly ask who is {character_a}'s {relation}
4. Sound natural and varied

Examples:
Original: "Who is David's mother?"
Paraphrased: "Can you tell me who David's mother is?"

Original: "Who is Sarah's child?"
Paraphrased: "I'd like to know who Sarah's child is."

Your response should contain ONLY the paraphrased question and nothing else."""
            
            completion = await self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            paraphrased_q = completion.choices[0].message.content.strip()
            # Clean any unnecessary quotes
            paraphrased_q = self._clean_quotes(paraphrased_q)
            
            new_q = question_data.copy()
            new_q["question"] = paraphrased_q
            return new_q
        except Exception as e:
            logger.error(f"Error paraphrasing question: {e}")
            return question_data
    
    def _clean_quotes(self, text: str) -> str:
        """Remove unnecessary quotes from a question string."""
        # Remove triple quotes
        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3]
        # Remove double quotes
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        # Remove single quotes
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        return text


class DatasetExporter:
    """Exports the generated data to CSV files."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def export_to_csv(self, forward_train_qs: List[Dict], forward_test_qs: List[Dict], backward_qs: List[Dict], output_dir: str):
        """Export forward train, forward test, and backward questions to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export forward training questions
        forward_train_file = os.path.join(output_dir, 'forward_train.csv')
        pd.DataFrame(forward_train_qs).to_csv(forward_train_file, index=False)
        logger.info(f"Forward training questions exported to {forward_train_file}")
        
        # Export forward testing questions
        forward_test_file = os.path.join(output_dir, 'forward_test.csv')
        pd.DataFrame(forward_test_qs).to_csv(forward_test_file, index=False)
        logger.info(f"Forward testing questions exported to {forward_test_file}")
        
        # Export backward questions
        backward_file = os.path.join(output_dir, 'backward.csv')
        pd.DataFrame(backward_qs).to_csv(backward_file, index=False)
        logger.info(f"Backward questions exported to {backward_file}")
        
        # Export graph as JSON
        graph_file = os.path.join(output_dir, 'relationship_graph.json')
        with open(graph_file, 'w') as f:
            json.dump(self.graph, f, indent=2)
        logger.info(f"Graph structure exported to {graph_file}")


def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


async def async_main():
    """Async main function to run the dataset generation process."""
    parser = argparse.ArgumentParser(description='Generate dataset for testing the reversal curse in LLMs')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config.get('output_dir', 'output')
    
    # Initialize components
    graph_generator = RelationshipGraphGenerator(config)
    question_generator = QuestionGenerator(config)
    exporter = DatasetExporter(config)
    
    # Generate relationship graph
    logger.info("Generating relationship graph...")
    graph = graph_generator.generate_graph()
    exporter.graph = graph  # Pass graph to exporter
    
    # Extract all relationships
    relationships = graph_generator.get_all_relationships()
    logger.info(f"Generated {len(relationships)} relationships between {len(graph_generator.characters)} characters")
    
    # Generate questions asynchronously
    logger.info("Generating questions...")
    forward_train_qs, forward_test_qs, backward_qs = await question_generator.generate_questions(relationships)
    
    # Always paraphrase the test set to make it different from the training set
    logger.info("Paraphrasing forward test questions for variety...")
    forward_test_qs = await question_generator.paraphrase_questions(forward_test_qs)
    
    # Optionally paraphrase the training set too if specified in config
    if config.get('paraphrase_questions', False):
        logger.info("Paraphrasing training and backward questions for variety...")
        forward_train_qs = await question_generator.paraphrase_questions(forward_train_qs)
        backward_qs = await question_generator.paraphrase_questions(backward_qs)
    
    # Export data
    logger.info("Exporting dataset...")
    exporter.export_to_csv(forward_train_qs, forward_test_qs, backward_qs, output_dir)
    
    if config.get('visualize_graph', False):
        logger.info("Generating graph visualization...")
        graph_generator.export_graph_visualization(output_dir)
    
    logger.info(f"Dataset generation complete. Results saved to {output_dir}")

def main():
    """Main function that runs the async main function."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
