#!/usr/bin/env python3

import re
import json
import os
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

# Load forward test data to have ground truth
forward_pairs = {}
with open("dataset/completions_sg/dataset/forward_test.csv", "r") as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            prompt = parts[0]
            expected = parts[1]
            # Extract the person and relation from prompt
            match = re.match(r"(.+)'s (.+) is", prompt)
            if match:
                person = match.group(1)
                relation = match.group(2)
                forward_pairs[person] = (expected, relation)

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

# Analyze errors
print(f"Total forward examples with errors: {len(errors)}")

# Check for errors related to training set characters
train_characters = set()

with open("dataset/completions_sg/dataset/training.csv", "r") as f:
    content = f.read()
    # Extract all character names from training data
    train_names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z']+(?:-[A-Z][a-z']+)?)\b", content)
    train_characters.update(train_names)

# Analyze if errors involve characters only in train set
print("\nErrors by character relationship to training set:")
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

print(f"Errors involving only characters in training set: {train_only_errors}")
print(f"Errors involving mix of train/test characters: {mixed_errors}")
print(f"Errors not involving any training characters: {other_errors}")

# Check for relation type confusion
relation_errors = defaultdict(list)
for error in errors:
    relation_errors[error["relation"]].append(error)

# Analyze if there's confusion between same relation types
relation_confusion = defaultdict(int)
for error in errors:
    person = error["person"]
    relation = error["relation"]
    expected = error["expected"]
    generated = clean_name(error["generated"])
    
    # Find if the generated output is valid for another person with same relation
    for other_person, (other_target, other_relation) in forward_relations.items():
        if other_person != person and other_relation == relation and other_target == generated:
            relation_confusion[relation] += 1
            print(f"\nRelation confusion in example {error['example_num']}:")
            print(f"  Original: {person}'s {relation} should be {expected}")
            print(f"  Generated: {generated}, which is actually {other_person}'s {relation}")

print("\nTotal relation type confusion errors:", sum(relation_confusion.values()))
for relation, count in relation_confusion.items():
    print(f"  {relation}: {count} confusion errors")

# Check for cases where the model generated a valid but wrong answer (where the generated answer 
# appears in the dataset as a valid target for some other source with the same relation)
valid_but_wrong = 0
for error in errors:
    person = error["person"]
    relation = error["relation"]
    generated = clean_name(error["generated"])
    
    # Check if generated is a valid target for any other source with same relation
    is_valid_elsewhere = False
    for other_person, (other_target, other_relation) in forward_relations.items():
        if other_person != person and other_relation == relation and other_target == generated:
            is_valid_elsewhere = True
            break
    
    if is_valid_elsewhere:
        valid_but_wrong += 1

print(f"\nCases where model generated a valid but wrong answer: {valid_but_wrong}")

# # Print all errors for inspection
# print("\nAll Forward Test Errors:")
# for i, error in enumerate(errors, 1):
#     print(f"{i}. {error['person']}'s {error['relation']} should be '{error['expected']}', but got '{error['generated']}'")