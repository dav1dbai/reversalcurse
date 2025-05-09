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
    def __init__(self, config: dict):
        self.config = config
        self.characters = []
        self.graph = {}
        self.nx_graph = nx.DiGraph()
        self.relations = []
        self.relation_map = {}
        try:
            with open(self.config['relations_file'], 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            bwd, fwd = parts[0], parts[1]
                            gender_tag = parts[2] if len(parts) > 2 else "n-n"
                            self.relations.append(tuple(parts))
                            self.relation_map[fwd] = (bwd, gender_tag)
                        else:
                            print(f"Warning: Skipping malformed relation line: {line.strip()}")
        except FileNotFoundError:
            print(f"Error: Relations file not found at {self.config['relations_file']}")
        except Exception as e:
            print(f"Error loading relations file: {e}")

    def generate_graph(self) -> Dict:
        character_file = self.config.get('character_names_file')
        all_characters = []
        character_genders = {}

        if not character_file or not os.path.exists(character_file):
             raise FileNotFoundError(f"Character names file not found or not specified: {character_file}")

        with open(character_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    name = parts[0]
                    gender = parts[1] if len(parts) > 1 and parts[1].strip() else 'n'
                    all_characters.append(name)
                    character_genders[name] = gender

        if not self.relations:
             print("Warning: No relations loaded. Graph will have no edges.")

        num_chars = min(self.config.get('num_characters', len(all_characters)), len(all_characters))

        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)

        self.characters = random.sample(all_characters, num_chars)
        adjacency_list = {char: [] for char in self.characters}
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(self.characters)

        min_rel = self.config.get('min_relationships', 1)
        max_rel = self.config.get('max_relationships', 1)
        gendered = self.config.get('gendered', True)
        enforce_exclusivity = (min_rel == 1 and max_rel == 1)

        print(f"Graph Generation Config: min_rel={min_rel}, max_rel={max_rel}, gendered={gendered}, exclusive={enforce_exclusivity}")

        if enforce_exclusivity:
            print("Using exclusive pairing logic (min_rel=1, max_rel=1).")
            available_characters = set(self.characters)

            while len(available_characters) >= 2:
                potential_sources = sorted(list(available_characters))
                random.shuffle(potential_sources)
                relationship_formed_this_iteration = False

                for source in potential_sources:
                    if source not in available_characters:
                        continue

                    source_gender = character_genders.get(source)
                    potential_targets_set = available_characters - {source}
                    sorted_potential_targets = sorted(list(potential_targets_set))
                    random.shuffle(sorted_potential_targets)

                    compatible_targets = []
                    for target in sorted_potential_targets:
                        target_gender = character_genders.get(target)
                        for rel_tuple in self.relations:
                             if len(rel_tuple) >= 2:
                                gender_tag = rel_tuple[2] if len(rel_tuple) > 2 else "n-n"
                                src_gender_match = True
                                tgt_gender_match = True
                                if gendered:
                                    src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                                    tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'

                                if src_gender_match and tgt_gender_match:
                                    compatible_targets.append(target)
                                    break

                    if compatible_targets:
                        target = random.choice(compatible_targets)
                        target_gender = character_genders.get(target)

                        compatible_relations_for_pair = []
                        for rel_tuple in self.relations:
                            if len(rel_tuple) >= 2:
                                gender_tag = rel_tuple[2] if len(rel_tuple) > 2 else "n-n"
                                src_gender_match = True
                                tgt_gender_match = True
                                if gendered:
                                    src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                                    tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'

                                if src_gender_match and tgt_gender_match:
                                    compatible_relations_for_pair.append(rel_tuple)

                        if compatible_relations_for_pair:
                            compatible_relations_for_pair.sort()
                            chosen_rel_tuple = random.choice(compatible_relations_for_pair)
                            forward_rel = chosen_rel_tuple[1]

                            adjacency_list[source].append({"target": target, "relation": forward_rel})
                            self.nx_graph.add_edge(source, target, relation=forward_rel)

                            available_characters.remove(source)
                            available_characters.remove(target)
                            relationship_formed_this_iteration = True
                            break

                if not relationship_formed_this_iteration:
                    break
        else:
            print(f"Using non-exclusive relationship logic (min_rel={min_rel}, max_rel={max_rel}).")
            potential_sources = sorted(list(self.characters))
            random.shuffle(potential_sources)

            for source in potential_sources:
                source_gender = character_genders.get(source)
                num_rels_to_add = random.randint(min_rel, max_rel)
                added_rels_count = 0

                potential_targets = sorted([c for c in self.characters if c != source])
                random.shuffle(potential_targets)

                for target in potential_targets:
                    if added_rels_count >= num_rels_to_add:
                        break

                    if source == target:
                        continue

                    if self.nx_graph.has_edge(source, target):
                        continue

                    target_gender = character_genders.get(target)

                    compatible_relations_for_pair = []
                    for rel_tuple in self.relations:
                        if len(rel_tuple) >= 2:
                            gender_tag = rel_tuple[2] if len(rel_tuple) > 2 else "n-n"
                            src_gender_match = True
                            tgt_gender_match = True
                            if gendered:
                                src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                                tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'

                            if src_gender_match and tgt_gender_match:
                                compatible_relations_for_pair.append(rel_tuple)

                    if compatible_relations_for_pair:
                        compatible_relations_for_pair.sort()
                        chosen_rel_tuple = random.choice(compatible_relations_for_pair)
                        forward_rel = chosen_rel_tuple[1]

                        adjacency_list[source].append({"target": target, "relation": forward_rel})
                        self.nx_graph.add_edge(source, target, relation=forward_rel)
                        added_rels_count += 1

        self.graph = {
            "characters": self.characters,
            "adjacency_list": adjacency_list
        }

        return self.graph
    
    def get_all_relationships(self) -> List[Dict]:
        relationships = []
        for source, targets in self.graph.get("adjacency_list", {}).items():
            for edge in targets:
                forward_relation = edge["relation"]
                target = edge["target"]
                if forward_relation in self.relation_map:
                    backward_relation, _ = self.relation_map[forward_relation]
                    relationships.append({
                        "character_a": source,
                        "character_b": target,
                        "forward_relation": forward_relation,
                        "backward_relation": backward_relation
                    })
                else:
                    print(f"Warning: Forward relation '{forward_relation}' from {source} to {target} not found in relation_map. Cannot determine backward relation.")
                    relationships.append({
                        "character_a": source,
                        "character_b": target,
                        "forward_relation": forward_relation,
                        "backward_relation": "unknown" 
                    })
        return relationships

    def export_graph_visualization(self, output_dir: str):
        if not self.nx_graph.nodes:
            print("Graph is empty. Skipping visualization.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping graph visualization.")
            return
        
        output_path = os.path.join(output_dir, "graph.png")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.nx_graph, k=0.3, iterations=50, seed=self.config.get('random_seed'))
        nx.draw(self.nx_graph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
        edge_labels = nx.get_edge_attributes(self.nx_graph, 'relation')
        nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Character Relationship Graph")
        plt.savefig(output_path)
        plt.close()
        print(f"Graph visualization saved to {output_path}")

    def export_to_csv(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        relationships = self.get_all_relationships()
        if not relationships:
            print(f"No relationships to export to CSV: {output_path}")
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["character_a", "character_b", "forward_relation", "backward_relation"]) 
            return

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ["character_a", "character_b", "forward_relation", "backward_relation"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(relationships)
        print(f"Relationships exported to {output_path}")

    def generate_test_files(self, output_dir):
        os.makedirs(os.path.join(output_dir, "dataset"), exist_ok=True)
        relationships = self.get_all_relationships()
        forward_test_items = []
        backward_test_items = []
        completion_test_items = []
        qa_pattern = self.config.get('qa_pattern', "Question: {question} Answer: {answer}")
        completion_prompt_format = self.config.get('completion_prompt_format', "{subject}'s {relation} is")

        for rel in relationships:
            fwd_q = qa_pattern.format(question=f"Who is {rel['character_a']}'s {rel['forward_relation']}?", answer=rel['character_b'])
            bwd_q = qa_pattern.format(question=f"Who is {rel['character_b']}'s {rel['backward_relation']}?", answer=rel['character_a'])
            forward_test_items.append({"text": fwd_q})
            backward_test_items.append({"text": bwd_q})
            
            completion_test_items.append({
                "prompt": completion_prompt_format.format(subject=rel['character_a'], relation=rel['forward_relation']),
                "expected_completion": rel['character_b']
            })
            completion_test_items.append({
                "prompt": completion_prompt_format.format(subject=rel['character_b'], relation=rel['backward_relation']),
                "expected_completion": rel['character_a']
            })

        def write_csv(filepath, items, headers):
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for item in items:
                    if isinstance(item, dict): # Ensure item is a dict
                         writer.writerow(item)
                    else:
                         print(f"Warning: Skipping non-dictionary item: {item} in {filepath}")

        write_csv(os.path.join(output_dir, "dataset", "forward_test.csv"), forward_test_items, ["text"])
        write_csv(os.path.join(output_dir, "dataset", "backward_test.csv"), backward_test_items, ["text"])
        write_csv(os.path.join(output_dir, "dataset", "completion_test.csv"), completion_test_items, ["prompt", "expected_completion"])
        print(f"Test files (forward, backward, completion) generated in {os.path.join(output_dir, 'dataset')}")

    async def generate_training_data(self, output_dir, api_key=None, num_paraphrases=3, model="gpt-4o"):
        output_path = os.path.join(output_dir, "dataset", "training.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not found. Skipping paraphrase generation.")
            self.generate_basic_training_data(output_path) 
            return

        client = AsyncOpenAI(api_key=api_key)
        relationships = self.get_all_relationships()
        all_training_items = [] 

        if self.config.get('dataset_type', 'qa') == 'completions':
            print("Generating training data for 'completions' type...")
            completion_prompt_format = self.config.get('completion_prompt_format', "{subject}'s {relation} is {object}")
            tasks = []
            for rel in relationships:
                source_a, target_a, relation_a = rel['character_a'], rel['character_b'], rel['forward_relation']
                source_b, target_b, relation_b = rel['character_b'], rel['character_a'], rel['backward_relation']
                
                tasks.append(self._generate_completion_prompts(client, source_a, target_a, relation_a, num_paraphrases, model))
                tasks.append(self._generate_completion_prompts(client, source_b, target_b, relation_b, num_paraphrases, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    original_rel_index = i // 2
                    is_forward = i % 2 == 0
                    original_rel = relationships[original_rel_index]
                    c_a = original_rel['character_a'] if is_forward else original_rel['character_b']
                    c_b = original_rel['character_b'] if is_forward else original_rel['character_a']
                    r = original_rel['forward_relation'] if is_forward else original_rel['backward_relation']
                    print(f"Error generating completion prompts for {c_a}-{r}-{c_b}: {result}")
                    all_training_items.append({"text": completion_prompt_format.format(subject=c_a, relation=r, object=c_b)})
                elif result:
                    for prompt_text in result:
                        all_training_items.append({"text": prompt_text})
                else: # Fallback if no paraphrases generated but no explicit error
                    original_rel_index = i // 2
                    is_forward = i % 2 == 0
                    original_rel = relationships[original_rel_index]
                    c_a = original_rel['character_a'] if is_forward else original_rel['character_b']
                    c_b = original_rel['character_b'] if is_forward else original_rel['character_a']
                    r = original_rel['forward_relation'] if is_forward else original_rel['backward_relation']
                    all_training_items.append({"text": completion_prompt_format.format(subject=c_a, relation=r, object=c_b)})
            
            headers = ["text"]
            def get_row_data(item):
                 return item # Item is already a dict like {"text": ...}

        elif self.config.get('dataset_type', 'qa') == 'qa':
            print("Generating training data for 'qa' type...")
            tasks = []
            for rel in relationships:
                q_fwd = f"Who is {rel['character_a']}'s {rel['forward_relation']}?"
                a_fwd = rel['character_b']
                q_bwd = f"Who is {rel['character_b']}'s {rel['backward_relation']}?"
                a_bwd = rel['character_a']
                tasks.append(self._generate_paraphrases(client, q_fwd, a_fwd, num_paraphrases, model))
                tasks.append(self._generate_paraphrases(client, q_bwd, a_bwd, num_paraphrases, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                original_rel_index = i // 2 
                is_forward = i % 2 == 0 
                original_question = f"Who is {relationships[original_rel_index]['character_a']}'s {relationships[original_rel_index]['forward_relation']}?" if is_forward else f"Who is {relationships[original_rel_index]['character_b']}'s {relationships[original_rel_index]['backward_relation']}?"
                original_answer = relationships[original_rel_index]['character_b'] if is_forward else relationships[original_rel_index]['character_a']

                if isinstance(result, Exception):
                    print(f"Error generating paraphrases for Q: {original_question} A: {original_answer}. Error: {result}")
                    all_training_items.append({"question": original_question, "answer": original_answer}) 
                elif result:
                    all_training_items.extend(result)
                else: 
                    all_training_items.append({"question": original_question, "answer": original_answer})
            
            headers = ["question", "answer"]
            def get_row_data(item):
                 return item # Item is already a dict like {"question": ..., "answer": ...}
        else:
            print(f"Unsupported dataset_type: {self.config.get('dataset_type')}. Defaulting to basic QA generation without API.")
            self.generate_basic_training_data(output_path)
            return

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for item in all_training_items:
                writer.writerow(get_row_data(item))
        print(f"Generated {len(all_training_items)} training examples (including paraphrases if API was used) and saved to {output_path}")

    async def _generate_paraphrases(self, client, question, answer, num_paraphrases=3, model="gpt-3.5-turbo"):
        paraphrases = []
        prompt = f"Generate {num_paraphrases} diverse ways to ask the following question, keeping the answer the same. " \
                 f"For each, provide the paraphrased question and the original answer.\n\n" \
                 f"Original Question: {question}\n" \
                 f"Original Answer: {answer}\n\n" \
                 f"Output each paraphrase as a new 'Question: ...' followed by 'Answer: ...' on the next line. " \
                 f"Do not number them or add extra text."
        
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in rephrasing questions for AI training."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            generated_text = completion.choices[0].message.content.strip()
            
            lines = generated_text.split('\n')
            i = 0
            while i < len(lines):
                if lines[i].startswith("Question:") and i + 1 < len(lines) and lines[i+1].startswith("Answer:"):
                    q = lines[i][len("Question:"):].strip()
                    a = lines[i+1][len("Answer:"):].strip()
                    if q and a == answer: # Ensure answer remains consistent
                        paraphrases.append({"question": q, "answer": a})
                    i += 2
                else:
                    i += 1
            
            if not paraphrases: # Fallback if parsing fails or no valid paraphrases
                 paraphrases.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"Error during OpenAI call for paraphrasing Q: '{question}': {e}")
            paraphrases.append({"question": question, "answer": answer}) # Fallback on error
        return paraphrases

    async def _generate_completion_prompts(self, client, source, target, relation, num_prompts=3, model="gpt-4o"):
        prompts = []
        base_statement = f"{source}'s {relation} is {target}."
        
        system_prompt = "You are an expert in creating diverse training prompts for language models. " \
                        "Your task is to rephrase the given statement in various ways that a model could complete. " \
                        "The rephrased prompt should lead to the original target when completed by the model."
        
        user_prompt = f"Given the statement: \"{base_statement}\"\n\n" \
                      f"Generate {num_prompts} diverse prompts where the original statement is a natural completion. " \
                      f"The prompts should be in a format like '[Subject]'s [Relation] is [Object].' or similar variations. " \
                      f"Each generated prompt must be a full sentence or statement that directly reflects the original relationship. " \
                      f"Focus on rephrasing the structure while preserving the core meaning and the subject-relation-object format.\n\n" \
                      f"Examples of good rephrased prompts for 'A's relation to B is C':\n" \
                      f"- A's connection to B, known as relation, is C.\n" \
                      f"- The relation that A has with B is specifically C.\n" \
                      f"- C is what A's relation to B is.\n\n" \
                      f"Output ONLY the rephrased full statements, each on a new line. Do not include any extra text, numbering, or explanations."

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=1024 
            )
            generated_text = completion.choices[0].message.content.strip()
            raw_prompts = [p.strip() for p in generated_text.split('\n') if p.strip()]
            
            for p_text in raw_prompts:
                cleaned_prompt = re.sub(r'^\d+[\.\)]\s*', '', p_text) # Remove leading numbers/bullets
                if cleaned_prompt: 
                    prompts.append(cleaned_prompt)

            if not prompts: # Fallback if parsing fails or no valid prompts
                 prompts.append(base_statement)
        except Exception as e:
            print(f"Error during OpenAI call for completion prompts for '{base_statement}': {e}")
            prompts.append(base_statement) # Fallback on error
        return prompts
    
    def generate_basic_training_data(self, output_path):
        print(f"Generating basic training data (no API paraphrasing) and saving to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        relationships = self.get_all_relationships()
        training_items = []

        if self.config.get('dataset_type', 'qa') == 'completions':
            completion_prompt_format = self.config.get('completion_prompt_format', "{subject}'s {relation} is {object}")
            for rel in relationships:
                training_items.append({"text": completion_prompt_format.format(subject=rel['character_a'], relation=rel['forward_relation'], object=rel['character_b'])})
                training_items.append({"text": completion_prompt_format.format(subject=rel['character_b'], relation=rel['backward_relation'], object=rel['character_a'])})
            headers = ["text"]
            data_to_write = training_items
        elif self.config.get('dataset_type', 'qa') == 'qa':
            qa_pattern = self.config.get('qa_pattern', "Question: {question} Answer: {answer}")
            for rel in relationships:
                fwd_q = f"Who is {rel['character_a']}'s {rel['forward_relation']}?"
                fwd_a = rel['character_b']
                bwd_q = f"Who is {rel['character_b']}'s {rel['backward_relation']}?"
                bwd_a = rel['character_a']
                training_items.append({"question": fwd_q, "answer": fwd_a})
                training_items.append({"question": bwd_q, "answer": bwd_a})
            headers = ["question", "answer"]
            data_to_write = training_items
        else:
            print(f"Unsupported dataset_type: {self.config.get('dataset_type')} for basic generation. No training file created.")
            return

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_to_write)
        print(f"Basic training data saved to {output_path} with {len(data_to_write)} items.")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def generate_training_with_api(config, output_dir):
    api_key = os.getenv("OPENAI_API_KEY")
    num_paraphrases = config.get('num_paraphrases_openai', 3)
    model = config.get('openai_model', "gpt-4o")
    
    generator = RelationshipGraphGenerator(config)
    graph_data = generator.generate_graph()
    output_json_path = os.path.join(output_dir, "relationships.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(graph_data, f, indent=4)
    print(f"Relationship graph data saved to {output_json_path}")

    generator.export_graph_visualization(output_dir)
    generator.export_to_csv(os.path.join(output_dir, "relationships.csv"))
    generator.generate_test_files(output_dir)
    
    await generator.generate_training_data(output_dir, api_key, num_paraphrases, model)

async def main():
    parser = argparse.ArgumentParser(description="Generate a character relationship graph and associated datasets.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    parser.add_argument("output_dir", help="Directory to save the generated files.")
    parser.add_argument("--no_api", action="store_true", help="Generate basic training data without calling OpenAI API for paraphrasing.")
    args = parser.parse_args()

    config_data = load_config(args.config)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    else:
        print(f"Output directory {args.output_dir} already exists. Files may be overwritten.")

    if args.no_api:
        print("API calls disabled. Generating basic training data.")
        generator = RelationshipGraphGenerator(config_data)
        graph_data = generator.generate_graph()
        output_json_path = os.path.join(args.output_dir, "relationships.json")
        with open(output_json_path, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"Relationship graph data saved to {output_json_path}")
        generator.export_graph_visualization(args.output_dir)
        generator.export_to_csv(os.path.join(args.output_dir, "relationships.csv"))
        generator.generate_test_files(args.output_dir)
        generator.generate_basic_training_data(os.path.join(args.output_dir, "dataset", "training.csv"))
    else:
        print("API calls enabled for paraphrasing training data.")
        await generate_training_with_api(config_data, args.output_dir)

if __name__ == "__main__":
    asyncio.run(main())