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
with open("dataset/completions_sn/relationships.json", "r") as f:
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
with open("dataset/completions_sn/dataset/training.csv", "r") as f:
    content = f.read()
    # Extract all character names from training data
    train_names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z']+(?:-[A-Z][a-z']+)?)\b", content)
    train_characters.update(train_names)

# Regular expression to extract results
result_pattern = re.compile(r"Example (\d+):\nPrompt: (.+)'s (.+) is\nExpected Completion: (.+)\nGenerated Completion: (.+)\nCorrect: (True|False)")

# Parse backward results from log file
backward_errors = []
with open("logs/qwen7b_1024_comp_snaug_completions_sn_completion_results.txt", "r") as f:
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

# 1. IN-DOMAIN ENTITY ERROR ANALYSIS
# In-Domain Entity Error: The proportion of the erroneous character answers that were characters in the training dataset
in_domain_entity_errors = 0

for error in backward_errors:
    generated_in_train = clean_name(error["generated"]) in train_characters
    if generated_in_train:
        in_domain_entity_errors += 1

print("\n1. IN-DOMAIN ENTITY ERROR ANALYSIS:")
if len(backward_errors) > 0:
    in_domain_error_rate = in_domain_entity_errors / len(backward_errors) * 100
    print(f"In-Domain Entity Error Rate: {in_domain_entity_errors}/{len(backward_errors)} ({in_domain_error_rate:.1f}%)")
    print(f"These are errors where the model generated a character that exists in the training set.")
else:
    print("No errors found to analyze.")

# Store error counts by relation type (for later analysis)
inv_relation_errors = defaultdict(list)
for error in backward_errors:
    inv_relation_errors[error["inv_relation"]].append(error)

# Check for most common incorrect responses
generated_responses = Counter([clean_name(error["generated"]) for error in backward_errors])
print("\nMost common incorrect responses in backward direction:")
if generated_responses:
    for response, count in generated_responses.most_common(10):
        if count > 1:
            print(f"{response}: {count} occurrences")
else:
    print("No incorrect responses found.")

# 2. RELATION-PRESERVING ERROR ANALYSIS
# Relation-Preserving Error: The proportion of erroneous character answers 
# that answered with a character that has the same relationship assignment
relation_preserving_errors = []
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
            relation_preserving_errors.append({
                "example_num": error["example_num"],
                "target": target,
                "inv_relation": inv_relation,
                "expected": expected,
                "generated": generated,
                "confused_with": other_error["target"]
            })
            break  # Only count each error once

print("\n2. RELATION-PRESERVING ERROR ANALYSIS:")
if len(backward_errors) > 0:
    relation_preserving_rate = len(relation_preserving_errors) / len(backward_errors) * 100
    print(f"Relation-Preserving Error Rate: {len(relation_preserving_errors)}/{len(backward_errors)} ({relation_preserving_rate:.1f}%)")
    print(f"These are errors where the model generated a character that has the same relationship type with someone else.")
    
    # List examples if there are any
    if relation_preserving_errors:
        print("\nExamples of relation-preserving errors:")
        for i, conf in enumerate(relation_preserving_errors[:5]):  # Limit to 5 examples
            print(f"  {conf['target']}'s {conf['inv_relation']} → \n  Expected: {conf['expected']}, \n  Generated: {conf['generated']} (which is {conf['confused_with']}'s {conf['inv_relation']})")
else:
    print("No errors found to analyze relation preservation.")

# 3. PROPORTION OF ERRORS BY RELATION TYPE
# Load relationship pairs from CSV
relationship_pairs = []
relation_counts = Counter()
relation_pair_counts = Counter()
relation_to_pairs = defaultdict(list)
character_relation_pairs = defaultdict(set)

with open("dataset/completions_sn/relationships.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Track the forward and backward relations
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

for error in backward_errors:
    target = error["target"]
    inv_relation = error["inv_relation"]
    expected = error["expected"]
    
    relation_errors[inv_relation].append(error)
    
    # Find the pair relationship if it exists
    for row in csv.DictReader(open("dataset/completions_sn/relationships.csv", "r")):
        if row["character_b"] == target and row["backward_relation"] == inv_relation:
            relation_pair = (row["forward_relation"], row["backward_relation"])
            relation_pair_errors[relation_pair].append(error)
            break

# Calculate error rates by inverse relation type
print("\nError rates by inverse relation type:")
# Sort with a safe division (handle case where relation_counts[r] is 0)
for inv_rel in sorted(relation_errors.keys(), 
                    key=lambda r: len(relation_errors[r])/relation_counts[r] if relation_counts[r] > 0 else float('inf'), 
                    reverse=True):
    total_rel_count = relation_counts[inv_rel]
    rel_errors = len(relation_errors[inv_rel])
    if total_rel_count > 0:
        error_rate = (rel_errors / total_rel_count) * 100
        print(f"{inv_rel}: {rel_errors}/{total_rel_count} total errors ({error_rate:.2f}%)")
    else:
        print(f"{inv_rel}: {rel_errors}/0 total errors (N/A - relation not in training set)")

print("\nError rates by relation pairs:")
# Print all pairs, including those with zero errors
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