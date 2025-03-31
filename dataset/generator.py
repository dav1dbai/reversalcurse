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
import re

dotenv.load_dotenv()

class RelationshipGraphGenerator:
    """Generates a directed graph of character relationships."""
    
    def __init__(self, config: dict):
        self.config = config
        self.characters = []
        self.graph = {}
        self.nx_graph = nx.DiGraph()

        # Load relations and create a lookup map during initialization
        self.relations = []  # Format: [(backward_rel, forward_rel, gender_tag), ...]
        self.relation_map = {} # Maps forward_relation -> (backward_relation, gender_tag)
        try:
            with open(self.config['relations_file'], 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            bwd, fwd = parts[0], parts[1]
                            gender_tag = parts[2] if len(parts) > 2 else "n-n"
                            self.relations.append(tuple(parts)) # Store original tuple
                            self.relation_map[fwd] = (bwd, gender_tag)
                        else:
                            print(f"Warning: Skipping malformed relation line: {line.strip()}")
        except FileNotFoundError:
            print(f"Error: Relations file not found at {self.config['relations_file']}")
            # Decide how to handle this - raise error, exit, or continue with empty relations?
            # For now, we'll continue, but graph generation will likely fail or be empty.
        except Exception as e:
            print(f"Error loading relations file: {e}")

    def generate_graph(self) -> Dict:
        """Generate a relationship graph based on the configuration."""
        # Load characters with their gender tags
        character_file = self.config.get('character_names_file')
        all_characters = []
        character_genders = {}
        
        with open(character_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    name = parts[0]
                    gender = parts[1] if len(parts) > 1 else 'n'
                    all_characters.append(name)
                    character_genders[name] = gender
        
        # Relations are loaded in __init__
        
        # Sample characters
        num_chars = min(self.config.get('num_characters', len(all_characters)), len(all_characters))
        
        # Set random seed if provided
        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        
        self.characters = random.sample(all_characters, num_chars)
        adjacency_list = {char: [] for char in self.characters}
        self.nx_graph = nx.DiGraph() # Initialize NetworkX graph
        
        # --- Start of Graph Generation Logic ---
        # This loop attempts to pair up available characters uniquely based on
        # compatible gender rules defined in the relations file.
        # Once a pair is formed, both characters are removed from the pool.
        available_characters = set(self.characters) # Characters available to form a relationship
        
        while len(available_characters) >= 2:
            # Convert set to list AND SORT IT before shuffling for reproducibility
            potential_sources = sorted(list(available_characters))
            random.shuffle(potential_sources) # Now shuffles a consistently ordered list
            relationship_formed_this_iteration = False
            
            for source in potential_sources:
                source_gender = character_genders.get(source)
                potential_targets_set = available_characters - {source}
                
                # Find targets compatible with this source based on gender rules
                compatible_targets = []
                # SORT potential_targets before iterating to ensure consistent order
                sorted_potential_targets = sorted(list(potential_targets_set))
                for target in sorted_potential_targets: # Iterate over the sorted list
                    target_gender = character_genders.get(target)
                    # Check against all defined relations for compatibility
                    for rel_tuple in self.relations:
                        if len(rel_tuple) >= 3:
                            gender_tag = rel_tuple[2]
                            # Check if source gender matches rule (or rule is neutral 'n')
                            src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                            # Check if target gender matches rule (or rule is neutral 'n')
                            tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'
                            if src_gender_match and tgt_gender_match:
                                compatible_targets.append(target)
                                break # Found a compatible relation, target is compatible
                
                if compatible_targets:
                    # Choose a random target from the compatible ones
                    # Sort compatible_targets before choosing for determinism/debugging.
                    compatible_targets.sort()
                    target = random.choice(compatible_targets)
                    target_gender = character_genders.get(target)
                    
                    # Find all relations compatible with this specific source-target pair
                    compatible_relations_for_pair = []
                    for rel_tuple in self.relations:
                        if len(rel_tuple) >= 3:
                            gender_tag = rel_tuple[2]
                            src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                            tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'
                            if src_gender_match and tgt_gender_match:
                                compatible_relations_for_pair.append(rel_tuple)
                    
                    # Choose a random relation from the compatible ones for this pair
                    # Sort compatible_relations_for_pair before choosing for determinism
                    compatible_relations_for_pair.sort() # Sort based on tuple comparison
                    chosen_rel_tuple = random.choice(compatible_relations_for_pair)
                    # relations tuple format: (backward_rel, forward_rel, gender_tag)
                    forward_rel = chosen_rel_tuple[1]
                    
                    # Add the relationship
                    adjacency_list[source].append({
                        "target": target, 
                        "relation": forward_rel
                    })
                    self.nx_graph.add_edge(source, target, relation=forward_rel)
                    
                    # Remove BOTH source and target from further consideration
                    available_characters.remove(source)
                    available_characters.remove(target)
                    relationship_formed_this_iteration = True
                    break # Move to the next iteration of the while loop
            
            # If no relationship could be formed in this pass with any available source, stop.
            if not relationship_formed_this_iteration:
                break
        # --- End of Graph Generation Logic ---
        
        self.graph = {
            "characters": self.characters,
            "adjacency_list": adjacency_list
        }
        
        # Add nodes to NetworkX graph for potentially isolated characters
        for char in self.characters:
            if char not in self.nx_graph:
                self.nx_graph.add_node(char)
        
        return self.graph
    
    def get_all_relationships(self) -> List[Dict]:
        """Extract all relationships from the graph using the pre-built map."""
        relationships = []
        # Use the relation_map created in __init__
        for source, targets in self.graph.get("adjacency_list", {}).items():
            for edge in targets:
                forward_relation = edge["relation"]
                target = edge["target"]
                # Look up backward relation and gender tag using the map
                backward_rel, gender_tag = self.relation_map.get(forward_relation, ("unknown", "n-n"))
                relationships.append({
                    "character_a": source,
                    "character_b": target,
                    "forward_relation": forward_relation,
                    "backward_relation": backward_rel,
                    "gender_tag": gender_tag # Use gender tag from the map
                })
        
        return relationships
    
    def export_graph_visualization(self, output_dir: str):
        """Export a visualization of the graph in DOT format."""
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Relationship Graph')
            
            for char in self.characters:
                dot.node(char)
            
            # Use the adjacency list from the generated graph
            for source, targets in self.graph.get("adjacency_list", {}).items():
                for edge in targets:
                    dot.edge(source, edge["target"], label=edge["relation"])
            
            output_path = os.path.join(output_dir, 'relationship_graph')
            try:
                 dot.render(output_path, format='png', cleanup=True)
                 print(f"Graph visualization saved to {output_path}.png")
            except Exception as e:
                 print(f"Failed to render graph visualization: {e}")
                 print("Ensure Graphviz executables are in your system's PATH.")

        except ImportError:
            print("Graphviz Python package not installed. Skipping visualization export.")
            print("Install with: pip install graphviz")
        except graphviz.backend.execute.ExecutableNotFound:
             print("Graphviz executable not found. Skipping visualization export.")
             print("Ensure Graphviz (the application, not just the Python package) is installed and in your system's PATH.")
            
    def export_to_csv(self, output_path):
        """Export relationships to CSV."""
        relationships = self.get_all_relationships() # Get formatted relationships
        if not relationships:
            print("No relationships found to export to CSV.")
            return

        with open(output_path, 'w', newline='') as csvfile:
            # Use keys from the first relationship dict for header (assumes consistency)
            header = relationships[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=header)

            writer.writeheader()
            writer.writerows(relationships)
        print(f"Exported {len(relationships)} relationships to {output_path}")
    
    def generate_test_files(self, output_dir):
        """Generate forward and backward test files based on relationships."""
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)

        training_format = self.config.get('training_data_format', 'qa') # Check format
        all_relationships = self.get_all_relationships() # Use the helper method

        if not all_relationships:
            print("No relationships found in the graph. Cannot generate test files.")
            return 0, 0

        forward_items = []
        backward_items = []

        for rel in all_relationships:
            source = rel['character_a']
            target = rel['character_b']
            forward_relation = rel['forward_relation']
            backward_relation = rel['backward_relation'] # Directly from get_all_relationships

            if training_format == 'completion':
                forward_items.append({'prompt': f"{source}'s {forward_relation} is", 'completion': target})
                backward_items.append({'prompt': f"{target}'s {backward_relation} is", 'completion': source})
            else: # Default QA format
                forward_items.append({'question': f"Who is {source}'s {forward_relation}?", 'answer': target})
                backward_items.append({'question': f"Who is {target}'s {backward_relation}?", 'answer': source})

        # Helper function to write CSV to avoid repetition
        def write_csv(filepath, items, headers):
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(items)
                print(f"Written {len(items)} examples to {filepath}")
            except Exception as e:
                 print(f"Error writing to {filepath}: {e}")

        # Define headers based on format
        if training_format == 'completion':
            headers = ['prompt', 'completion']
        else:
            headers = ['question', 'answer']

        # Write forward test file
        forward_file = os.path.join(dataset_dir, 'forward_test.csv')
        write_csv(forward_file, forward_items, headers)

        # Write backward test file
        backward_file = os.path.join(dataset_dir, 'backward_test.csv')
        write_csv(backward_file, backward_items, headers)

        print(f"Generated forward and backward test examples in '{training_format}' format.")
        return len(forward_items), len(backward_items)

    async def generate_training_data(self, output_dir, api_key=None, num_paraphrases=3, model="gpt-4o"):
        """Generate training data (QA or single-column Completion) for FORWARD relationships using OpenAI."""
        # Ensure the dataset directory exists
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)

        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)

        # Determine training data format from config
        training_format = self.config.get('training_data_format', 'qa') # Default to 'qa'

        # --- Read relationships directly from relationships.csv ---
        relationships = []
        relationships_file_path = os.path.join(output_dir, 'relationships.csv')

        try:
            with open(relationships_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure all necessary columns exist and are not empty
                    if all(row.get(key) for key in ['character_a', 'character_b', 'forward_relation', 'backward_relation']):
                        relationships.append({
                            "source": row['character_a'],
                            "target": row['character_b'],
                            "forward_relation": row['forward_relation'],
                            # "backward_relation": row['backward_relation'] # No longer strictly needed here
                        })
                    else:
                         print(f"Warning: Skipping incomplete relationship row: {row}")
        except FileNotFoundError:
            print(f"Error: {relationships_file_path} not found. Cannot generate training data.")
            return 0
        except Exception as e:
            print(f"Error reading {relationships_file_path}: {e}")
            return 0

        if not relationships:
            print("No relationships found or read from relationships.csv. Skipping training data generation.")
            return 0

        # --- Generate Training Data (Forward Relationships Only) ---
        training_items = []
        batch_size = self.config.get('api_batch_size', 10) # Configurable batch size
        total_processed = 0

        print(f"Generating {num_paraphrases} examples per FORWARD relationship for {len(relationships)} relationships in '{training_format}' format using model '{model}'...")

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            tasks = []

            for rel in batch:
                if training_format == 'completion':
                    # Generate prompts ONLY for the forward relation (completion = target)
                    # Use the full num_paraphrases count for the forward relation
                    if num_paraphrases > 0:
                        tasks.append(self._generate_completion_prompts(
                            client, rel['source'], rel['target'], rel['forward_relation'], num_paraphrases, model
                        ))
                else: # Default to 'qa' format
                    # Generate paraphrased question ONLY for the forward relation
                    question = f"Who is {rel['source']}'s {rel['forward_relation']}?"
                    answer = rel['target']
                    if num_paraphrases > 0:
                        tasks.append(self._generate_paraphrases(
                            client, question, answer, num_paraphrases, model
                        ))

            # Gather results - helper functions return [] on error
            results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions from tasks

            # Process results, extending training_items
            for item_list in results:
                if isinstance(item_list, Exception):
                    print(f"Warning: An error occurred during API call: {item_list}")
                elif isinstance(item_list, list):
                    training_items.extend(item_list) # extend handles empty lists correctly
                # else: ignore unexpected types

            total_processed += len(batch)
            print(f"Processed {total_processed}/{len(relationships)} relationships")

        # --- Write training data to CSV ---
        training_file = os.path.join(dataset_dir, 'training.csv')
        written_count = 0

        # Define headers and row extraction logic based on format
        if training_format == 'completion':
            headers = ['text']
            def get_row_data(item):
                if isinstance(item, dict) and 'prompt' in item and 'completion' in item:
                    # Combine prompt and completion into a single string
                    return {'text': f"{item['prompt']} {item['completion']}"}
                return None
        else: # QA format
            headers = ['question', 'answer']
            def get_row_data(item):
                 if isinstance(item, dict) and 'question' in item and 'answer' in item:
                     return item # Already in correct dict format for DictWriter
                 return None

        try:
            with open(training_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for item in training_items:
                    row_data = get_row_data(item)
                    if row_data:
                        writer.writerow(row_data)
                        written_count += 1
                    else:
                         print(f"Warning: Skipping malformed {training_format} item for training file: {item}")
            print(f"Successfully wrote {written_count} training examples to {training_file}")
        except Exception as e:
             print(f"Error writing training file {training_file}: {e}")
             return 0 # Indicate failure

        return written_count # Return the actual number written

    async def _generate_paraphrases(self, client, question, answer, num_paraphrases=3, model="gpt-3.5-turbo"):
        """Generate paraphrases for a single question using OpenAI (QA format)."""
        prompt = f"""
        Please generate {num_paraphrases} different paraphrases of the following question.
        Keep the same meaning but vary the wording, syntax, and phrasing.
        Only output the paraphrased questions, one per line, with no additional text.
        Do not include numbering like "1." or any other prefixes.
        Do not enclose the paraphrased questions in quotation marks.

        Original question: {question}
        """

        try:
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

            # Clean any list numbers and remove surrounding quotes
            cleaned_paraphrases = []
            for p in paraphrases:
                # Remove list numbers/bullets first
                cleaned = re.sub(r'^\s*(\d+[\.\)\]:]|[\-\*•])\s*', '', p)
                # Remove leading/trailing quotes (single or double)
                if len(cleaned) >= 2 and cleaned.startswith(('"', "'")) and cleaned.endswith(('"', "'")):
                    cleaned = cleaned[1:-1]
                cleaned_paraphrases.append(cleaned)

            # Ensure we don't exceed the requested number of paraphrases
            cleaned_paraphrases = cleaned_paraphrases[:num_paraphrases]

            # Create question-answer pairs
            qa_pairs = [{'question': p, 'answer': answer} for p in cleaned_paraphrases]
            return qa_pairs
        except Exception as e:
            print(f"Error generating paraphrases for '{question}': {e}")
            return []

    async def _generate_completion_prompts(self, client, source, target, relation, num_prompts=3, model="gpt-4o"):
        """Generate sentence prompts for a relationship using OpenAI (Completion format)."""
        # The 'target' is the expected completion. The prompt should end just before it.
        prompt = f"""
Your task is to generate {num_prompts} different sentence completion prompts based on the specific relationship: '{source}' is the '{relation}' of '{target}'.

Each generated prompt MUST satisfy these conditions:
1.  It must contain the name '{source}'.
2.  It must contain the relationship term '{relation}'.
3.  It must be phrased as an incomplete sentence or fragment.
4.  The only correct word or name needed to complete the prompt is exactly '{target}'.
5.  The prompt's text must end immediately before where '{target}' would naturally be placed.

Think of different ways to express the connection using both '{source}' and '{relation}'. For instance:
- "{source}'s {relation} is "
- "The individual who serves as the {relation} for {source} is named "
- "Regarding {source}, their {relation} is known to be "
- "We know that {source} has a {relation}, who is "

Output Instructions:
- Generate ONLY the prompts themselves.
- Each prompt should be on a new line.
- Do NOT include any numbering (like 1., 2.), bullet points, quotation marks around the prompts, or the completion word '{target}' in your output.

Relationship Details:
- Source Person: {source}
- Relationship Type: {relation}
- Target Person (Completion): {target}
"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates sentence completion prompts based on relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8, # Slightly higher temp for more variation
                max_tokens=1024
            )

            result = response.choices[0].message.content.strip()
            prompts = [line.strip() for line in result.split('\n') if line.strip()]

            # Clean any list numbers or prefixes and surrounding quotes
            cleaned_prompts = []
            for p in prompts:
                # Remove list numbers/bullets first
                # Ensure the prompt doesn't accidentally end with the target name already
                if not p.endswith(target):
                     cleaned_prompts.append(p)
                else:
                    print(f"Warning: Skipping generated prompt ending with target: '{p}' for target '{target}'")


            # Ensure we don't exceed the requested number
            cleaned_prompts = cleaned_prompts[:num_prompts]

            # Create prompt-completion pairs
            prompt_completion_pairs = [{'prompt': p, 'completion': target} for p in cleaned_prompts]
            return prompt_completion_pairs
        except Exception as e:
            print(f"Error generating completion prompts for '{source}' -> '{relation}' -> '{target}': {e}")
            return []

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

async def generate_training_with_api(config, output_dir):
    """Generate training data using the OpenAI API."""
    # Get API key and other parameters from config
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        # Consider raising an exception or returning a specific error code
        # instead of just printing and returning 0.
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Use 'num_paraphrases' config key for both QA and completion formats
    num_examples_per_relationship = config.get('num_paraphrases', 3)
    # Get model from config, defaulting to gpt-4o-mini if not specified
    model = config.get('openai_model', 'gpt-4o') # Default model set here

    # Initialize generator - relations are loaded here
    try:
        generator = RelationshipGraphGenerator(config)
    except Exception as e:
        print(f"Error initializing RelationshipGraphGenerator: {e}")
        # Decide on error handling: re-raise, return error code, etc.
        raise # Re-raise the exception for now

    # Generate training data
    # Pass the model explicitly obtained from config
    print(f"Starting training data generation with model: {model}") # Added log
    try:
        num_training_examples = await generator.generate_training_data(
            output_dir, api_key, num_examples_per_relationship, model
        )
        print(f"Generated {num_training_examples} training examples.")
        return num_training_examples
    except Exception as e:
        print(f"An error occurred during training data generation: {e}")
        # Decide on error handling
        raise # Re-raise the exception

async def main():
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
    try:
        generator = RelationshipGraphGenerator(config)
        print("Generating relationship graph...")
        graph = generator.generate_graph()
        if not graph:
             print("Graph generation failed or produced an empty graph. Exiting.")
             return # Exit if graph generation failed

        # Export results
        output_prefix = os.path.join(config['output_dir'], 'relationships')
        print(f"Exporting relationships to {output_prefix}.csv...")
        generator.export_to_csv(f"{output_prefix}.csv")

        print("Attempting to export graph visualization...")
        generator.export_graph_visualization(config['output_dir'])

        # Generate test files
        print("Generating test files...")
        forward_count, backward_count = generator.generate_test_files(config['output_dir'])
        # Print counts only if generation was successful (implied by reaching here)
        # The function itself prints details now.

        # Save graph as JSON for reference
        print(f"Saving graph structure to {output_prefix}.json...")
        with open(f"{output_prefix}.json", 'w') as f:
            json.dump(graph, f, indent=2)

        # Generate training data if requested
        if args.training or config.get('generate_training', False):
            print("Generating training data with OpenAI API...")
            # generate_training_with_api now raises exceptions on failure
            await generate_training_with_api(config, config['output_dir'])

        print(f"Generation complete. Results saved to {config['output_dir']}")

    except FileNotFoundError as e:
         print(f"Error: A required file was not found: {e}")
    except ValueError as e: # Catch specific errors like missing API key
         print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Optionally add more detailed error logging here
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())