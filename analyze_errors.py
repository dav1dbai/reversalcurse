import re
import json
import os
import csv
from collections import Counter, defaultdict

def clean_name(name):
    return re.sub(r"^(named|known as) ", "", name)

with open("dataset/completions_sn/relationships.json", "r") as f:
    relationships_data = json.load(f)

characters = set(relationships_data["characters"])
forward_relations = {}
reverse_relations = {}

for person, relations in relationships_data["adjacency_list"].items():
    for relation in relations:
        target = relation["target"]
        rel_type = relation["relation"]
        forward_relations[person] = (target, rel_type)
        
        if target not in reverse_relations:
            reverse_relations[target] = []
        reverse_relations[target].append((person, rel_type))

train_characters = set()
with open("dataset/completions_sn/dataset/training.csv", "r") as f:
    content = f.read()
    train_names = re.findall(r"\\b([A-Z][a-z]+ [A-Z][a-z']+(?:-[A-Z][a-z']+)?)\\b", content)
    train_characters.update(train_names)

result_pattern = re.compile(r"Example (\\d+):\\nPrompt: (.+)'s (.+) is\\nExpected Completion: (.+)\\nGenerated Completion: (.+)\\nCorrect: (True|False)")

errors = []
with open("logs/qwen7b_1024_comp_snaug_completions_sn_completion_results.txt", "r") as f:
    content = f.read()
    forward_section = content.split("===== FORWARD TEST RESULTS =====")[1].split("===== BACKWARD TEST RESULTS =====")[0]
    
    for match in result_pattern.finditer(forward_section):
        example_num = int(match.group(1))
        person = match.group(2)
        relation = match.group(3)
        expected = match.group(4)
        generated = match.group(5)
        correct = match.group(6) == "True"
        
        if not correct:
            errors.append({
                "example_num": example_num,
                "person": person,
                "relation": relation,
                "expected": expected,
                "generated": generated,
                "correct": correct
            })

print(f"Total forward examples with errors: {len(errors)}")

in_domain_entity_errors = 0

for error in errors:
    generated_in_train = clean_name(error["generated"]) in train_characters
    if generated_in_train:
        in_domain_entity_errors += 1

print("\\n1. IN-DOMAIN ENTITY ERROR ANALYSIS:")
if len(errors) > 0:
    in_domain_error_rate = in_domain_entity_errors / len(errors) * 100
    print(f"In-Domain Entity Error Rate: {in_domain_entity_errors}/{len(errors)} ({in_domain_error_rate:.1f}%)")
    print(f"These are errors where the model generated a character that exists in the training set.")
else:
    print("No errors found to analyze.")

relation_preserving_errors = []
for error in errors:
    person = error["person"]
    relation = error["relation"]
    expected = error["expected"]
    generated = clean_name(error["generated"])
    
    for other_person, (other_target, other_relation) in forward_relations.items():
        if other_person != person and other_relation == relation and other_target == generated:
            relation_preserving_errors.append({
                "example_num": error["example_num"],
                "person": person,
                "relation": relation,
                "expected": expected,
                "generated": generated,
                "confused_with": other_person
            })
            break

print("\\n2. RELATION-PRESERVING ERROR ANALYSIS:")
if len(errors) > 0:
    relation_preserving_rate = len(relation_preserving_errors) / len(errors) * 100
    print(f"Relation-Preserving Error Rate: {len(relation_preserving_errors)}/{len(errors)} ({relation_preserving_rate:.1f}%)")
    print(f"These are errors where the model generated a character that has the same relationship type with someone else.")
    
    if relation_preserving_errors:
        print("\\nExamples of relation-preserving errors:")
        for i, conf in enumerate(relation_preserving_errors[:5]):
            print(f"  {conf['person']}'s {conf['relation']} → \\n  Expected: {conf['expected']}, \\n  Generated: {conf['generated']} (which is {conf['confused_with']}'s {conf['relation']})")
else:
    print("No errors found to analyze relation preservation.")

confusion_by_relation = defaultdict(list)
for conf in relation_preserving_errors:
    confusion_by_relation[conf["relation"]].append(conf)

if confusion_by_relation:
    for relation, confusions in sorted(confusion_by_relation.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\\n{relation} confusion: {len(confusions)} errors")
        for conf in confusions:
            print(f"  Example {conf['example_num']}: {conf['person']}'s {conf['relation']} → Generated: {conf['generated']}, which is {conf['confused_with']}'s {conf['relation']}")
else:
    print("\\nNo relation confusion examples found.")

print("\\nExamples of in-domain entity errors (both entities in training set):")
in_domain_examples = [error for error in errors if error["person"] in train_characters and clean_name(error["generated"]) in train_characters]
if in_domain_examples:
    for i, error in enumerate(in_domain_examples[:5]):
        print(f"  {error['person']}'s {error['relation']} → \\n  Expected: {error['expected']}, \\n  Generated: {error['generated']}")
else:
    print("  No in-domain entity errors found.")

relationship_pairs = []
relation_counts = Counter()
relation_pair_counts = Counter()
relation_to_pairs = defaultdict(list)
character_relation_pairs = defaultdict(set)

with open("dataset/completions_sn/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        forward_relation = row["forward_relation"]
        backward_relation = row["backward_relation"]
        char_a = row["character_a"]
        char_b = row["character_b"]
        
        relation_counts[forward_relation] += 1
        relation_counts[backward_relation] += 1
        
        relation_pair = (forward_relation, backward_relation)
        relation_pair_counts[relation_pair] += 1
        
        relation_to_pairs[forward_relation].append((char_a, char_b))
        relation_to_pairs[backward_relation].append((char_b, char_a))
        
        character_relation_pairs[(char_a, forward_relation)].add(char_b)
        character_relation_pairs[(char_b, backward_relation)].add(char_a)

relation_errors = defaultdict(list)
relation_pair_errors = defaultdict(list)

for error in errors:
    person = error["person"]
    relation = error["relation"]
    expected = error["expected"]
    
    relation_errors[relation].append(error)
    
    for row in csv.DictReader(open("dataset/completions_sn/relationships.csv", "r")):
        if row["character_a"] == person and row["forward_relation"] == relation:
            relation_pair = (row["forward_relation"], row["backward_relation"])
            relation_pair_errors[relation_pair].append(error)
            break
        elif row["character_b"] == person and row["backward_relation"] == relation:
            relation_pair = (row["backward_relation"], row["forward_relation"])
            relation_pair_errors[relation_pair].append(error)
            break

print("\\nError rates by relation type:")
for rel in sorted(relation_errors.keys(), 
                  key=lambda r: len(relation_errors[r])/relation_counts[r] if relation_counts[r] > 0 else float('inf'), 
                  reverse=True):
    total_count = relation_counts[rel]
    errors_count = len(relation_errors[rel])
    if total_count > 0:
        error_rate = (errors_count / total_count) * 100
        print(f"{rel}: {errors_count}/{total_count} total errors ({error_rate:.2f}%)")
    else:
        print(f"{rel}: {errors_count}/0 total errors (N/A - relation not in training set)")

print("\\nError rates by relation pairs:")
for pair in sorted(relation_pair_counts.keys(), 
                  key=lambda p: (len(relation_pair_errors.get(p, []))/relation_pair_counts[p] 
                               if p in relation_pair_errors and relation_pair_counts[p] > 0 
                               else 0), 
                  reverse=True):
    forward_rel, backward_rel = pair
    total_count = relation_pair_counts[pair]
    errors_count = len(relation_pair_errors.get(pair, []))
    error_rate = (errors_count / total_count) * 100 if total_count > 0 else 0
    print(f"{forward_rel}/{backward_rel}: {errors_count}/{total_count} total errors ({error_rate:.2f}%)")