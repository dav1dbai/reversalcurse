from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm  # Import tqdm for progress bars
import os
import argparse  # Add argparse import

# Add argument parsing
parser = argparse.ArgumentParser(description="Run inference with optional chat templating.")
parser.add_argument(
    '--no_chat_template',
    action='store_true',
    help="If set, do not use the tokenizer's chat template for formatting. Use a basic format instead."
)
parser.add_argument(
    '--model_path',
    type=str,
    required=True,
    help="Path to the fine-tuned model or LoRA adapter directory."
)
parser.add_argument(
    '--base_model_name',
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Base model name from Hugging Face Hub (used if loading LoRA adapters)."
)
args = parser.parse_args()
is_chat_format = not args.no_chat_template  # True if --no_chat_template is NOT used

print("Loading data...")
dataset_name = "qa_sn_aug"
forward_test_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/forward_test.csv')
forward_train_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/training.csv')
backward_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/backward_test.csv')

# Use arguments for model paths
model_path = args.model_path
model_name = args.base_model_name  # Base model for LoRA

# Construct log_file_path dynamically
model_name_for_log = os.path.basename(model_path.rstrip('/')) # e.g., qwen7b_512_qacn
log_file_path = f"../logs/{model_name_for_log}_generation_results.txt"

def format_data(df):
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            "question": row["question"],
            "answer": row["answer"]
        })
    return formatted_data

train_data = format_data(forward_train_df)
forward_test_data = format_data(forward_test_df)
backward_test_data = format_data(backward_df)

train_dataset = Dataset.from_list(train_data)
forward_test_dataset = Dataset.from_list(forward_test_data)
backward_test_dataset = Dataset.from_list(backward_test_data)


print("Loading model...")
# Define model paths

# Choose whether to load a full fine-tuned model or a LoRA model
use_full_model = False  # Set to True for full fine-tuned model, False for LoRA

if use_full_model:
    # Load full fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  # Path to your full fine-tuned model
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("Loaded full fine-tuned model")
else:
    # Load base model and attach LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,  # Original base model
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and attach LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        model_path,  # Path to your saved adapter weights
        is_trainable=False,  # Set to False for inference
        local_files_only=True
    )
    print("Loaded LoRA model")

# Explicitly move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print device information to confirm
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU! This will be very slow.")

# After loading the model, add these diagnostic prints:
print("Model type:", type(model))
print("Is PeftModel:", isinstance(model, PeftModel))
if hasattr(model, "active_adapter"):
    print("Active adapter:", model.active_adapter)
if hasattr(model, "peft_config"):
    print("PEFT config:", model.peft_config)

# 4. Evaluate reversal curse
print("Evaluating reversal curse...")

def evaluate_dataset(dataset, batch_size=32):
    correct = 0
    total = len(dataset)
    predictions = []
    
    # Enable Flash Attention if available
    if hasattr(model, "config"):
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"
    
    # Process in batches with tqdm progress bar
    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating")
    
    for i in progress_bar:
        batch = dataset[i:min(i+batch_size, total)]
        
        questions = batch["question"]
        true_answers = batch["answer"]
        
        # Format the questions based on chat template flag
        if is_chat_format:
            prompts = [
                tokenizer.apply_chat_template([
                    {"role": "user", "content": question},
                ], tokenize=False, add_generation_prompt=True)
                for question in questions
            ]
        else:
            # Use a basic format if chat template is disabled
            prompts = [
                f"USER: {question}\nASSISTANT:" 
                for question in questions
            ]
        
        # print(prompts)
        # Tokenize the batch
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding="longest",
            padding_side="left",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Process each example in the batch
        for j, (output, true_answer) in enumerate(zip(outputs, true_answers)):
            # Get the exact input length without relying on padding
            input_length = len(inputs.input_ids[j])
            
            # Debug prints to understand the tokenization (temporary)
            if i == 0 and j < 2:  # Just for the first couple of examples
                print(f"Input prompt: '{prompts[j]}'")
                print(f"Input tokens length: {input_length}")
                print(f"Full output: '{tokenizer.decode(output, skip_special_tokens=True)}'")
            
            # Decode only the newly generated tokens
            prediction = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            
            # Clean up any lingering artifacts
            # if prediction.startswith(":"):
            #     prediction = prediction[1:].strip()
            
            predictions.append(prediction)
            
            is_correct = true_answer.lower() == prediction.lower()
            if is_correct:
                correct += 1
            
            if i == 0 and j < 5:
                print(f"Question: {questions[j]}")
                print(f"True answer: {true_answer}")
                print(f"Prediction: {prediction}")
                print(f"Correct: {is_correct}\n")
        
        # Report batch performance and update progress bar
        progress_bar.set_postfix({
            "acc": f"{(correct/(len(predictions)))*100:.2f}%", 
        })
    
    accuracy = (correct / total) * 100
    return accuracy, predictions

# Evaluate on forward test set
print("\nEvaluating on forward test set...")
forward_accuracy, forward_predictions = evaluate_dataset(forward_test_dataset)
print(f"Forward accuracy: {forward_accuracy:.2f}%")

# Evaluate on backward test set
print("\nEvaluating on backward test set...")
backward_accuracy, backward_predictions = evaluate_dataset(backward_test_dataset)
print(f"Backward accuracy: {backward_accuracy:.2f}%")

# Print the summary
print("\nReversal Curse Results:")
print(f"Forward accuracy: {forward_accuracy:.2f}%")
print(f"Backward accuracy: {backward_accuracy:.2f}%")

if forward_accuracy > 0:
    print(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}")
    print(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%")

# Log all generations to a text file
print("Saving generations to log file...")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

with open(log_file_path, "w") as f:
    f.write("===== REVERSAL CURSE EVALUATION RESULTS =====\n\n")
    f.write(f"Forward accuracy: {forward_accuracy:.2f}%\n")
    f.write(f"Backward accuracy: {backward_accuracy:.2f}%\n")
    if forward_accuracy > 0:
        f.write(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}\n")
        f.write(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%\n\n")
    
    # Log forward test results
    f.write("===== FORWARD TEST RESULTS =====\n")
    for i, (question, answer, prediction) in enumerate(zip(forward_test_dataset["question"], 
                                                          forward_test_dataset["answer"], 
                                                          forward_predictions)):
        f.write(f"Example {i+1}:\n")
        f.write(f"Question: {question}\n")
        f.write(f"True answer: {answer}\n")
        f.write(f"Prediction: {prediction}\n")
        is_correct = answer.lower() == prediction.lower()
        f.write(f"Correct: {is_correct}\n\n")
    
    # Log backward test results
    f.write("===== BACKWARD TEST RESULTS =====\n")
    for i, (question, answer, prediction) in enumerate(zip(backward_test_dataset["question"], 
                                                          backward_test_dataset["answer"], 
                                                          backward_predictions)):
        f.write(f"Example {i+1}:\n")
        f.write(f"Question: {question}\n")
        f.write(f"True answer: {answer}\n")
        f.write(f"Prediction: {prediction}\n")
        is_correct = answer.lower() == prediction.lower()
        f.write(f"Correct: {is_correct}\n\n")

print(f"Generations saved to {log_file_path}")
print("Done!")