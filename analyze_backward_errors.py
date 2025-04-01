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
reverse_relations = defaultdict(list)  # target -> [(person, relation), ...]

# Define inverse relations mapping
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

for person, relations in relationships_data["adjacency_list"].items():
    for relation in relations:
        target = relation["target"]
        rel_type = relation["relation"]
        forward_relations[person] = (target, rel_type)
        
        # Create reverse mapping
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

print(f"Total backward examples with errors: {len(backward_errors)}")

# 1. PATTERN IN ERRORS - CHARACTERS FROM TRAINING SET
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

# 2. RELATION TYPE CONFUSION
inv_relation_errors = defaultdict(list)
for error in backward_errors:
    inv_relation_errors[error["inv_relation"]].append(error)

print("\nBackward errors by inverse relation type:")
for inv_rel, errors in sorted(inv_relation_errors.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{inv_rel}: {len(errors)} errors")

# Check for most common incorrect responses
generated_responses = Counter([clean_name(error["generated"]) for error in backward_errors])
print("\nMost common incorrect responses in backward direction:")
for response, count in generated_responses.most_common(10):
    if count > 1:
        print(f"{response}: {count} occurrences")

# Check if there's confusion in the backward direction
backward_confusion = 0
confusion_examples = []
for error in backward_errors:
    target = error["target"]
    inv_relation = error["inv_relation"]
    expected = error["expected"]
    generated = clean_name(error["generated"])
    
    # Check if the generated output is actually a valid answer for a different question
    # with the same relation type
    for other_error in backward_errors:
        if (other_error["example_num"] != error["example_num"] and 
            other_error["inv_relation"] == inv_relation and
            other_error["expected"] == generated):
            backward_confusion += 1
            confusion_examples.append({
                "example_num": error["example_num"],
                "target": target,
                "inv_relation": inv_relation,
                "expected": expected,
                "generated": generated,
                "confused_with": other_error["target"]
            })
            print(f"\nBackward confusion in example {error['example_num']}:")
            print(f"  Original: {target}'s {inv_relation} should be {expected}")
            print(f"  Generated: {generated}, which is actually {other_error['target']}'s {inv_relation}")
            break  # Only count each error once

print(f"\nTotal backward relation confusion errors: {backward_confusion}")

# 3. PROPORTION OF ERRORS BY RELATION TYPE
# Count relationship types in the entire dataset
backward_relation_counts = Counter()
with open("dataset/completions_sg/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        backward_relation_counts[row["backward_relation"]] += 1

# Calculate error rates by inverse relation type
print("\nError rates by inverse relation type:")
for inv_rel in sorted(inv_relation_errors.keys(), key=lambda r: len(inv_relation_errors[r])/backward_relation_counts[r], reverse=True):
    total_rel_count = backward_relation_counts[inv_rel]
    rel_errors = len(inv_relation_errors[inv_rel])
    error_rate = (rel_errors / total_rel_count) * 100
    print(f"{inv_rel}: {rel_errors}/{total_rel_count} total errors ({error_rate:.2f}%)")