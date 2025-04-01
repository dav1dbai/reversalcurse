#!/usr/bin/env python3

import re
import json
import os
import csv
from collections import Counter, defaultdict

def clean_name(name):
    """Clean names by removing prefixes like 'named' or 'known as'"""
    return re.sub(r"^(named|known as) ", "", name)

# Load relationships data
with open("dataset/completions_sg/relationships.json", "r") as f:
    relationships_data = json.load(f)

# Extract characters and their relationships
characters = set(relationships_data["characters"])
forward_relations = {}  # person -> (target, relation)
reverse_relations = {}  # target -> (person, relation)

for person, relations in relationships_data["adjacency_list"].items():
    for relation in relations:
        target = relation["target"]
        rel_type = relation["relation"]
        forward_relations[person] = (target, rel_type)
        
        # Create reverse mapping
        if target not in reverse_relations:
            reverse_relations[target] = []
        reverse_relations[target].append((person, rel_type))

# Load training data to identify characters in training set
train_characters = set()
with open("dataset/completions_sg/dataset/training.csv", "r") as f:
    content = f.read()
    # Extract all character names from training data
    train_names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z']+(?:-[A-Z][a-z']+)?)\b", content)
    train_characters.update(train_names)

# Regular expression to extract results
result_pattern = re.compile(r"Example (\d+):\nPrompt: (.+)'s (.+) is\nExpected Completion: (.+)\nGenerated Completion: (.+)\nCorrect: (True|False)")

# Parse results from log file
errors = []
with open("logs/completion_generation_results_reventity.txt", "r") as f:
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

# 1. PATTERN IN ERRORS - CHARACTERS FROM TRAINING SET
train_only_errors = 0
mixed_errors = 0
other_errors = 0

for error in errors:
    person_in_train = error["person"] in train_characters
    expected_in_train = error["expected"] in train_characters
    generated_in_train = clean_name(error["generated"]) in train_characters
    
    if person_in_train and expected_in_train and generated_in_train:
        train_only_errors += 1
    elif person_in_train or expected_in_train or generated_in_train:
        mixed_errors += 1
    else:
        other_errors += 1

print("\n1. CHARACTER ANALYSIS:")
print(f"Errors involving only characters in training set: {train_only_errors}/{len(errors)} ({train_only_errors/len(errors)*100:.1f}%)")

# 2. RELATION TYPE CONFUSION
relation_confusion_examples = []
for error in errors:
    person = error["person"]
    relation = error["relation"]
    expected = error["expected"]
    generated = clean_name(error["generated"])
    
    # Find if the generated output is valid for another person with same relation
    for other_person, (other_target, other_relation) in forward_relations.items():
        if other_person != person and other_relation == relation and other_target == generated:
            relation_confusion_examples.append({
                "example_num": error["example_num"],
                "person": person,
                "relation": relation,
                "expected": expected,
                "generated": generated,
                "confused_with": other_person
            })
            break  # Only count each error once

print("\n2. RELATION CONFUSION ANALYSIS:")
print(f"Errors showing confusion between two instances of the same relation type: {len(relation_confusion_examples)}/{len(errors)} ({len(relation_confusion_examples)/len(errors)*100:.1f}%)")

# Group confusion by relation type
confusion_by_relation = defaultdict(list)
for conf in relation_confusion_examples:
    confusion_by_relation[conf["relation"]].append(conf)

# Print confusion details by relation type
for relation, confusions in sorted(confusion_by_relation.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n{relation} confusion: {len(confusions)} errors")
    for conf in confusions:
        print(f"  Example {conf['example_num']}: {conf['person']}'s {conf['relation']} → Generated: {conf['generated']}, which is {conf['confused_with']}'s {conf['relation']}")

# 3. PROPORTION OF ERRORS BY RELATION TYPE
# Load relationship pairs from CSV
relationship_pairs = []
relation_counts = Counter()
relation_pair_counts = Counter()
relation_to_pairs = defaultdict(list)
character_relation_pairs = defaultdict(set)

with open("dataset/completions_sg/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Track the forward relation
        forward_relation = row["forward_relation"]
        backward_relation = row["backward_relation"]
        char_a = row["character_a"]
        char_b = row["character_b"]
        
        # Count individual relations
        relation_counts[forward_relation] += 1
        relation_counts[backward_relation] += 1
        
        # Count relation pairs
        relation_pair = (forward_relation, backward_relation)
        relation_pair_counts[relation_pair] += 1
        
        # Track character pairs with specific relations
        relation_to_pairs[forward_relation].append((char_a, char_b))
        relation_to_pairs[backward_relation].append((char_b, char_a))
        
        # Track which characters have which relations to each other
        character_relation_pairs[(char_a, forward_relation)].add(char_b)
        character_relation_pairs[(char_b, backward_relation)].add(char_a)

# Count errors by relation type
relation_errors = defaultdict(list)
relation_pair_errors = defaultdict(list)

for error in errors:
    person = error["person"]
    relation = error["relation"]
    expected = error["expected"]
    
    relation_errors[relation].append(error)
    
    # Find the pair relationship if it exists
    for row in csv.DictReader(open("dataset/completions_sg/relationships.csv", "r")):
        if row["character_a"] == person and row["forward_relation"] == relation:
            relation_pair = (row["forward_relation"], row["backward_relation"])
            relation_pair_errors[relation_pair].append(error)
            break
        elif row["character_b"] == person and row["backward_relation"] == relation:
            relation_pair = (row["backward_relation"], row["forward_relation"])
            relation_pair_errors[relation_pair].append(error)
            break

print("\nError rates by relation type:")
for rel in sorted(relation_errors.keys(), key=lambda r: len(relation_errors[r])/relation_counts[r], reverse=True):
    total_count = relation_counts[rel]
    errors_count = len(relation_errors[rel])
    error_rate = (errors_count / total_count) * 100
    print(f"{rel}: {errors_count}/{total_count} total errors ({error_rate:.2f}%)")

print("\nError rates by relation pairs:")
# Print all pairs, including those with zero errors
for pair in sorted(relation_pair_counts.keys(), key=lambda p: len(relation_pair_errors.get(p, []))/relation_pair_counts[p] if p in relation_pair_errors else 0, reverse=True):
    forward_rel, backward_rel = pair
    total_count = relation_pair_counts[pair]
    errors_count = len(relation_pair_errors.get(pair, []))
    error_rate = (errors_count / total_count) * 100 if total_count > 0 else 0
    print(f"{forward_rel}/{backward_rel}: {errors_count}/{total_count} total errors ({error_rate:.2f}%)")
    
    # Uncomment these lines to see the specific relationships with errors
    # print("  Error details:")
    # for error in relation_errors[rel]:
    #     print(f"  - {error['person']}'s {rel} should be {error['expected']}, got {error['generated']}")