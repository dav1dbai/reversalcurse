import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from openai import OpenAI
import time

print("Loading data...")
forward_test_df = pd.read_csv('../dataset/output/synthetic/dataset/forward_test.csv')
backward_df = pd.read_csv('../dataset/output/synthetic/dataset/backward_test.csv')

def format_data(df):
    """Simple formatter for questions."""
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            "question": row["question"],
            "answer": row["answer"]
        })
    return formatted_data

# Just use backward questions directly without enhancement
backward_test_data = format_data(backward_df)

print("Initializing OpenAI client for GPT-4o...")
client = OpenAI()

# Evaluate function
def evaluate_dataset(dataset, batch_size=10, model="gpt-4o"):
    correct = 0
    total = len(dataset)
    predictions = []
    
    # Process in batches with tqdm progress bar
    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating")
    
    for i in progress_bar:
        batch = dataset[i:min(i+batch_size, total)]
        batch_predictions = []
        
        for item in batch:
            question = item["question"]
            true_answer = item["answer"]
            
            # Prompt to encourage reasoning about what information would be needed
            cot_question = f"""Answer this question: {question}

Think step by step:
1. What is this question asking about? Identify the entities and relationship.
2. What information would I need to answer this question accurately?
3. What would be the logical "forward" question I would need to know the answer to?
4. If I had that information, how would I use it to answer the original question?
5. Based on my best reasoning, what's the most likely answer?

Provide your reasoning and final answer."""
            
            try:
                # Get response from GPT-4o
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": cot_question}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                prediction = response.choices[0].message.content.strip()
                batch_predictions.append(prediction)
                
                # Check if the true answer is in the prediction
                is_correct = true_answer.lower() in prediction.lower()
                if is_correct:
                    correct += 1
                
                # Print some examples
                if i == 0 and len(batch_predictions) <= 5:
                    print(f"Question: {question}")
                    print(f"CoT Question: {cot_question}")
                    print(f"True answer: {true_answer}")
                    print(f"Prediction: {prediction}")
                    print(f"Correct: {is_correct}\n")
                    
            except Exception as e:
                print(f"Error processing question: {question}")
                print(f"Error: {e}")
                batch_predictions.append("Error")
                
        predictions.extend(batch_predictions)
        
        # Report batch performance and update progress bar
        progress_bar.set_postfix({
            "acc": f"{(correct/len(predictions))*100:.2f}%", 
        })
    
    accuracy = (correct / total) * 100
    return accuracy, predictions

# Evaluate on backward test set without forward information
print("\nEvaluating on backward test set with reasoning about needed information...")
backward_accuracy, backward_predictions = evaluate_dataset(backward_test_data)
print(f"Backward accuracy with reasoning approach: {backward_accuracy:.2f}%")

# Print the summary
print("\nChain of Thought Results with Reasoning Approach:")
print(f"Backward accuracy: {backward_accuracy:.2f}%")

print("Done!")