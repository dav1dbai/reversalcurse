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
with open("logs/completion_generation_results_sg.txt", "r") as f:
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

print(f"=== ANALYSIS OF FORWARD TEST ERRORS ({len(errors)} total errors) ===")

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
# Count relationship types in the entire dataset
relation_counts = Counter()
with open("dataset/completions_sg/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        relation_counts[row["forward_relation"]] += 1

# Count errors by relation type
relation_errors = defaultdict(list)
for error in errors:
    relation_errors[error["relation"]].append(error)

print("\nError rates by relation type:")
for rel in sorted(relation_errors.keys(), key=lambda r: len(relation_errors[r])/relation_counts[r], reverse=True):
    total_count = relation_counts[rel]
    errors_count = len(relation_errors[rel])
    error_rate = (errors_count / total_count) * 100
    confusion_count = len(confusion_by_relation.get(rel, []))
    print(f"{rel}: {errors_count}/{total_count} total errors ({error_rate:.2f}%)")