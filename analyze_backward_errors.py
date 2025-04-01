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
reverse_relations = defaultdict(list)  # target -> [(person, relation), ...]

for person, relations in relationships_data["adjacency_list"].items():
    for relation in relations:
        target = relation["target"]
        rel_type = relation["relation"]
        forward_relations[person] = (target, rel_type)
        
        # Create reverse mapping
        reverse_relations[target].append((person, rel_type))

# Define inverse relations
inverse_relations = {
    "patient": "doctor",
    "friend": "friend",
    "employee": "boss",
    "brother": "sister",
    "child": "father",
    "researcher": "professor",
    "grandchild": "grandparent",
    "sailor": "captain",
    "subject": "king",
    "student": "principal",
    "mentee": "mentor",
    "client": "therapist",
    "athlete": "coach",
    "passenger": "pilot",
    "nephew": "aunt"
}

# Load backward test data for reference
backward_pairs = {}
with open("dataset/completions_sg/dataset/backward_test.csv", "r") as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            prompt = parts[0]
            expected = parts[1]
            # Extract the target and inverse relation from prompt
            match = re.match(r"(.+)'s (.+) is", prompt)
            if match:
                target = match.group(1)
                inv_relation = match.group(2)
                backward_pairs[target] = (expected, inv_relation)

# Load training data to identify characters in training set
train_characters = set()
with open("dataset/completions_sg/dataset/training.csv", "r") as f:
    content = f.read()
    # Extract all character names from training data
    train_names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z']+(?:-[A-Z][a-z']+)?)\b", content)
    train_characters.update(train_names)

# Regular expression to extract results
result_pattern = re.compile(r"Example (\d+):\nPrompt: (.+)'s (.+) is\nExpected Completion: (.+)\nGenerated Completion: (.+)\nCorrect: (True|False)")

# Parse backward results from log file
backward_errors = []
with open("logs/completion_generation_results_sg.txt", "r") as f:
    content = f.read()
    backward_section = content.split("===== BACKWARD TEST RESULTS =====")[1]
    
    for match in result_pattern.finditer(backward_section):
        example_num = int(match.group(1))
        target = match.group(2)
        inv_relation = match.group(3)
        expected = match.group(4)
        generated = match.group(5)
        correct = match.group(6) == "True"
        
        if not correct:  # Only collect errors
            backward_errors.append({
                "example_num": example_num,
                "target": target,
                "inv_relation": inv_relation,
                "expected": expected,
                "generated": generated,
                "correct": correct
            })

# Analyze backward errors
print(f"Total backward examples with errors: {len(backward_errors)}")

# Check if errors involve training set characters
train_only_errors = 0
mixed_errors = 0
other_errors = 0

for error in backward_errors:
    target_in_train = error["target"] in train_characters
    expected_in_train = error["expected"] in train_characters
    generated_in_train = clean_name(error["generated"]) in train_characters
    
    if target_in_train and expected_in_train and generated_in_train:
        train_only_errors += 1
    elif target_in_train or expected_in_train or generated_in_train:
        mixed_errors += 1
    else:
        other_errors += 1

print("\nBackward errors by character relationship to training set:")
print(f"Errors involving only characters in training set: {train_only_errors}")
print(f"Errors involving mix of train/test characters: {mixed_errors}")
print(f"Errors not involving any training characters: {other_errors}")

# Check for error distribution by inverse relation type
inv_relation_errors = defaultdict(list)
for error in backward_errors:
    inv_relation_errors[error["inv_relation"]].append(error)

print("\nBackward errors by inverse relation type:")
for inv_rel, errors in sorted(inv_relation_errors.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{inv_rel}: {len(errors)} errors")

# Load CSV to count relationship types
import csv
backward_relation_counts = Counter()
with open("dataset/completions_sg/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        backward_relation_counts[row["backward_relation"]] += 1

# Calculate error rates by inverse relation type
print("\nError rates by inverse relation type:")
for inv_rel in sorted(backward_relation_counts.keys()):
    total_rel_count = backward_relation_counts[inv_rel]
    rel_errors = len(inv_relation_errors.get(inv_rel, []))
    if rel_errors > 0:  # Only show relations with errors
        error_rate = (rel_errors / total_rel_count) * 100
        print(f"{inv_rel}: {rel_errors}/{total_rel_count} total errors ({error_rate:.2f}%)")

# Check for most common incorrect responses
generated_responses = Counter([clean_name(error["generated"]) for error in backward_errors])
print("\nMost common incorrect responses in backward direction:")
for response, count in generated_responses.most_common(10):
    if count > 1:
        print(f"{response}: {count} occurrences")

# Check if there's confusion in the backward direction
# (where model outputs a character from correct relation type but wrong instance)
backward_confusion = 0
for error in backward_errors:
    target = error["target"]
    inv_relation = error["inv_relation"]
    expected = error["expected"]
    generated = clean_name(error["generated"])
    
    # Convert inverse relation to forward relation if possible
    forward_rel = None
    for rel, inv_rel in inverse_relations.items():
        if inv_rel == inv_relation:
            forward_rel = rel
            break
    
    if forward_rel:
        # Check if the generated answer is valid for this relation but different target
        for other_target in reverse_relations:
            if other_target != target:
                for person, rel_type in reverse_relations[other_target]:
                    if rel_type == forward_rel and person == generated:
                        backward_confusion += 1
                        print(f"\nBackward confusion in example {error['example_num']}:")
                        print(f"  Original: {target}'s {inv_relation} should be {expected}")
                        print(f"  Generated: {generated}, which is actually {other_target}'s {inv_relation}")
                        break

print(f"\nTotal backward relation confusion errors: {backward_confusion}")

# # Print sample of backward errors for inspection
# print("\nSample of backward test errors:")
# for i, error in enumerate(backward_errors[:20], 1):
#     target_in_train = error["target"] in train_characters
#     expected_in_train = error["expected"] in train_characters
#     generated_in_train = clean_name(error["generated"]) in train_characters
    
#     train_info = []
#     if target_in_train:
#         train_info.append("target")
#     if expected_in_train:
#         train_info.append("expected")
#     if generated_in_train:
#         train_info.append("generated")
    
#     train_status = f" [In train: {', '.join(train_info)}]" if train_info else ""
    
#     print(f"{i}. {error['target']}'s {error['inv_relation']} should be '{error['expected']}', but got '{error['generated']}'{train_status}")