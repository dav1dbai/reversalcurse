#!/usr/bin/env python3
import json
import networkx as nx
import os
import random
import yaml
import argparse
import asyncio
import csv
from typing import Dict, List
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

class RelationshipGraphGenerator:
    """Generates a directed graph of character relationships."""
    
    def __init__(self, config: dict):
        self.config = config
        self.characters = []
        self.relations = []
        self.graph = {}
        self.nx_graph = nx.DiGraph()
    
    def generate_graph(self) -> Dict:
        """Generate a relationship graph based on the configuration."""
        # Load characters
        character_file = self.config.get('character_names_file')
        with open(character_file, 'r') as f:
            all_characters = [line.strip() for line in f if line.strip()]
        
        # Load relations
        with open(self.config['relations_file'], 'r') as f:
            self.relations = [
                tuple(line.strip().split(',')) 
                for line in f if line.strip()
            ]
        
        # Sample characters
        num_chars = min(self.config['num_characters'], len(all_characters))
        
        # Set random seed if provided
        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        
        self.characters = random.sample(all_characters, num_chars)
        adjacency_list = {char: [] for char in self.characters}
        
        # Generate relationships
        min_rel = self.config.get('min_relationships')
        max_rel = self.config.get('max_relationships')
        
        for i, source in enumerate(self.characters):
            max_possible = min(max_rel, num_chars - i - 1)
            
            if max_possible < min_rel:
                continue
            
            num_relations = random.randint(min_rel, max_possible)
            available_targets = self.characters[i+1:]
            
            if len(available_targets) < num_relations:
                num_relations = len(available_targets)
            
            targets = random.sample(available_targets, num_relations)
            
            for target in targets:
                rel_idx = random.randint(0, len(self.relations) - 1)
                forward_rel = self.relations[rel_idx][0]
                adjacency_list[source].append({"target": target, "relation": forward_rel})
                self.nx_graph.add_edge(source, target, relation=forward_rel)
        
        self.graph = {
            "characters": self.characters,
            "adjacency_list": adjacency_list
        }
        
        return self.graph
    
    def get_all_relationships(self) -> List[Dict]:
        """Extract all relationships from the graph."""
        relationships = []
        relation_map = {fwd: bwd for fwd, bwd in self.relations}
        
        for source, targets in self.graph["adjacency_list"].items():
            for edge in targets:
                relationships.append({
                    "character_a": source,
                    "character_b": edge["target"],
                    "forward_relation": edge["relation"],
                    "backward_relation": relation_map.get(edge["relation"])
                })
        
        return relationships
    
    def export_graph_visualization(self, output_dir: str):
        """Export a visualization of the graph in DOT format."""
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Relationship Graph')
            
            for char in self.characters:
                dot.node(char)
            
            for source, targets in self.graph["adjacency_list"].items():
                for edge in targets:
                    dot.edge(source, edge["target"], label=edge["relation"])
            
            dot.render(os.path.join(output_dir, 'relationship_graph'), format='png')
        except ImportError:
            print("Graphviz not available. Skipping visualization export.")
            
    def export_to_csv(self, output_path):
        """Export relationships to CSV."""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['character_a', 'character_b', 'forward_relation', 'backward_relation'])
            
            for source, targets in self.graph['adjacency_list'].items():
                for target_info in targets:
                    target = target_info['target']
                    forward_relation = target_info['relation']
                    
                    # Find the backward relation
                    rel_idx = next((i for i, r in enumerate(self.relations) if r[0] == forward_relation), None)
                    backward_relation = self.relations[rel_idx][1] if rel_idx is not None else "unknown"
                    
                    writer.writerow([source, target, forward_relation, backward_relation])
    
    def generate_test_files(self, output_dir):
        """Generate forward and backward test files based on relationships."""
        # Ensure the dataset directory exists
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        forward_questions = []
        backward_questions = []
        
        # Process relationships
        for source, targets in self.graph['adjacency_list'].items():
            for target_info in targets:
                target = target_info['target']
                forward_relation = target_info['relation']
                
                # Find the backward relation
                rel_idx = next((i for i, r in enumerate(self.relations) if r[0] == forward_relation), None)
                backward_relation = self.relations[rel_idx][1] if rel_idx is not None else "unknown"
                
                # Create forward question
                forward_questions.append({
                    'question': f"Who is {source}'s {forward_relation}?",
                    'answer': target
                })
                
                # Create backward question
                backward_questions.append({
                    'question': f"Who is {target}'s {backward_relation}?", 
                    'answer': source
                })
        
        # Write forward test file
        forward_file = os.path.join(dataset_dir, 'forward_test.csv')
        with open(forward_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in forward_questions:
                writer.writerow([item['question'], item['answer']])
        
        # Write backward test file
        backward_file = os.path.join(dataset_dir, 'backward_test.csv')
        with open(backward_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in backward_questions:
                writer.writerow([item['question'], item['answer']])
        
        return len(forward_questions), len(backward_questions)
    
    async def generate_training_data(self, output_dir, api_key=None, num_paraphrases=3, model="gpt-3.5-turbo"):
        """Generate training data by paraphrasing test questions using OpenAI."""
        # Ensure the dataset directory exists
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Read forward test questions
        forward_test_path = os.path.join(dataset_dir, 'forward_test.csv')
        training_questions = []
        
        # Read the original questions and answers
        original_qa_pairs = []
        with open(forward_test_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    original_qa_pairs.append((row[0], row[1]))
        
        # Process each question in batches to avoid rate limits
        batch_size = 10
        total_processed = 0
        
        print(f"Generating {num_paraphrases} paraphrases for {len(original_qa_pairs)} questions...")
        
        for i in range(0, len(original_qa_pairs), batch_size):
            batch = original_qa_pairs[i:i+batch_size]
            tasks = []
            
            for question, answer in batch:
                task = self._generate_paraphrases(client, question, answer, num_paraphrases, model)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            for paraphrased_qa_pairs in results:
                training_questions.extend(paraphrased_qa_pairs)
            
            total_processed += len(batch)
            print(f"Processed {total_processed}/{len(original_qa_pairs)} questions")
        
        # Write training data to CSV
        training_file = os.path.join(dataset_dir, 'training.csv')
        with open(training_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in training_questions:
                writer.writerow([item['question'], item['answer']])
        
        return len(training_questions)
    
    async def _generate_paraphrases(self, client, question, answer, num_paraphrases, model):
        """Generate paraphrases for a single question using OpenAI."""
        prompt = f"""
        Please generate {num_paraphrases} different paraphrases of the following question.
        Keep the same meaning but vary the wording, syntax, and phrasing.
        Only output the paraphrased questions, one per line, with no additional text.
        
        Original question: {question}
        """
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content.strip()
        paraphrases = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Ensure we don't exceed the requested number of paraphrases
        paraphrases = paraphrases[:num_paraphrases]
        
        # Create question-answer pairs
        qa_pairs = [{'question': p, 'answer': answer} for p in paraphrases]
        return qa_pairs

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

async def generate_training_with_api(config, output_dir):
    """Generate training data using the OpenAI API."""
    # Get API key and other parameters from config
    api_key = os.environ.get('OPENAI_API_KEY')
    num_paraphrases = config.get('num_paraphrases', 3)
    model = config.get('openai_model', 'gpt-4o-mini')

    
    # Initialize generator
    generator = RelationshipGraphGenerator(config)
    
    # Generate training data
    num_training_examples = await generator.generate_training_data(
        output_dir, api_key, num_paraphrases, model
    )
    
    return num_training_examples

def main():
    """Main function to run the relationship graph generator."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a character relationship graph')
    parser.add_argument('--config', '-c', default='dataset/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    parser.add_argument('--training', '-t', action='store_true',
                        help='Generate training data using OpenAI API')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified in command line
    if args.output:
        config['output_dir'] = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize and run generator
    generator = RelationshipGraphGenerator(config)
    graph = generator.generate_graph()
    
    # Export results
    output_prefix = os.path.join(config['output_dir'], 'relationships')
    generator.export_to_csv(f"{output_prefix}.csv")
    generator.export_graph_visualization(config['output_dir'])
    
    # Generate test files
    forward_count, backward_count = generator.generate_test_files(config['output_dir'])
    print(f"Generated {forward_count} forward test questions and {backward_count} backward test questions")
    
    # Save graph as JSON for reference
    with open(f"{output_prefix}.json", 'w') as f:
        json.dump(graph, f, indent=2)
    
    # Generate training data if requested
    if args.training or config.get('generate_training', False):
        print("Generating training data with OpenAI API...")
        asyncio.run(generate_training_with_api(config, config['output_dir']))
    
    print(f"Generation complete. Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()