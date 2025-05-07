from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm  # Import tqdm for progress bars
import os
import argparse # Import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run inference for completion tasks with specified models and datasets.")
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
    help="Base model name from Hugging Face Hub (used if loading LoRA adapters). Default: Qwen/Qwen2.5-7B-Instruct"
)
parser.add_argument(
    '--dataset_name',
    type=str,
    default="completions_sn",
    help="Name of the dataset directory under ../dataset/ (e.g., completions_sg, completions_cn). Default: completions_sg"
)
args = parser.parse_args()
# --- End Argument Parsing ---

print("Loading data...")
# Use arguments for dataset paths
forward_test_df = pd.read_csv(f'../dataset/{args.dataset_name}/dataset/forward_test.csv')
backward_df = pd.read_csv(f'../dataset/{args.dataset_name}/dataset/backward_test.csv')

# Use arguments for model paths and log file name construction
base_model_for_lora = args.base_model_name # Renamed for clarity within LoRA loading block
model_adapter_path = args.model_path     # Renamed for clarity

model_basename = os.path.basename(args.model_path.rstrip('/'))
log_file_path = f"../logs/{model_basename}_{args.dataset_name}_completion_results.txt"

# Create datasets directly from pandas DataFrames
forward_test_dataset = Dataset.from_pandas(forward_test_df[['prompt', 'completion']])
backward_test_dataset = Dataset.from_pandas(backward_df[['prompt', 'completion']])

print(f"Forward test examples: {len(forward_test_dataset)}")
print(f"Backward test examples: {len(backward_test_dataset)}")
print("Example forward test datapoint:", forward_test_dataset[0])


print("Loading model...")
# Define model paths

# Choose whether to load a full fine-tuned model or a LoRA model
use_full_model = False # Set to True for full fine-tuned model, False for LoRA - Example: using LoRA

if use_full_model:
    # Load full fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        model_adapter_path,  # Use argument for model path
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    )
    tokenizer = AutoTokenizer.from_pretrained(model_adapter_path) # Use argument for model path
    print("Loaded full fine-tuned model")
else:
    # Load base model and attach LoRA weights
    print(f"Loading base model: {base_model_for_lora}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_for_lora,  # Use argument for base model name
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        trust_remote_code=True, # Added trust_remote_code
    )
    print(f"Loading tokenizer for: {base_model_for_lora}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_for_lora, # Use argument for base model name
        trust_remote_code=True, # Added trust_remote_code
    )
    # Set pad token if necessary (often needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # Load and attach LoRA weights
    print(f"Loading LoRA adapter from: {model_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        model_adapter_path,  # Use argument for adapter path
        is_trainable=False  # Set to False for inference
    )
    print("Loaded LoRA model")

# Explicitly move model to GPU (redundant if device_map='auto' worked, but safe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() # Set model to evaluation mode

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

# 4. Evaluate model on completion task
print("Evaluating completion performance...")

def evaluate_dataset(dataset, batch_size=16): # Reduced default batch size for potentially larger models/sequences
    correct = 0
    total = len(dataset)
    predictions = []
    results = [] # Store detailed results

    # Enable Flash Attention if available (Optional, requires flash-attn installed)
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Could not enable Flash Attention 2: {e}")
        model.config.attn_implementation = None # Fallback

    # Process in batches with tqdm progress bar
    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating")

    for i in progress_bar:
        batch = dataset[i:min(i+batch_size, total)]

        prompts = batch["prompt"]
        true_completions = batch["completion"]

        # Tokenize the batch of prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            padding_side="left", # Use left padding for batch generation
            truncation=True,
            max_length=1024 # Adjust based on your model's max length and expected prompt length
        ).to(model.device)

        # Estimate max_new_tokens needed (optional, can use a fixed large number too)
        # This is a heuristic, might need adjustment
        avg_completion_len = np.mean([len(tokenizer(c)['input_ids']) for c in true_completions])
        max_new_tokens_to_generate = int(avg_completion_len) + 20 # Add buffer

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_to_generate, # Generate enough tokens for completion
                temperature=0.1, # Low temperature for deterministic output
                # top_p=0.95, # Not needed if do_sample=False
                do_sample=False, # Use deterministic generation for evaluation
                pad_token_id=tokenizer.pad_token_id,
                # use_cache=True, # Cache is usually enabled by default
            )

        # Process each example in the batch
        for j, (output, prompt, true_completion) in enumerate(zip(outputs, prompts, true_completions)):
            input_length = inputs.input_ids[j].shape[0] # Get length of the specific input sequence

            # Decode only the newly generated tokens
            generated_ids = output[input_length:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            predictions.append(prediction)

            # Compare generated completion with the true completion
            is_correct = true_completion.lower().strip() in prediction.lower().strip()
            if is_correct:
                correct += 1

            # Store detailed result for logging
            results.append({
                "prompt": prompt,
                "expected_completion": true_completion,
                "generated_completion": prediction,
                "correct": is_correct
            })

            # Print first few examples for debugging
            if i == 0 and j < 5:
                print(f"Prompt: {prompt}")
                print(f"Expected Completion: {true_completion}")
                print(f"Generated Completion: {prediction}")
                print(f"Correct: {is_correct}\n")

        # Report batch performance and update progress bar
        current_accuracy = (correct / len(predictions)) * 100 if len(predictions) > 0 else 0
        progress_bar.set_postfix({"acc": f"{current_accuracy:.2f}%"})

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, results # Return detailed results along with accuracy

# Evaluate on forward test set
print("\nEvaluating on forward test set...")
forward_accuracy, forward_results = evaluate_dataset(forward_test_dataset)
print(f"Forward accuracy: {forward_accuracy:.2f}%")

# Evaluate on backward test set
print("\nEvaluating on backward test set...")
backward_accuracy, backward_results = evaluate_dataset(backward_test_dataset)
print(f"Backward accuracy: {backward_accuracy:.2f}%")

# Print the summary
print("\nCompletion Task Results:")
print(f"Forward accuracy: {forward_accuracy:.2f}%")
print(f"Backward accuracy: {backward_accuracy:.2f}%")

if forward_accuracy > 0:
    print(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}")
    print(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%")
else:
    print("Cannot calculate ratio/drop due to zero forward accuracy.")


# Log all generations to a text file
print(f"Saving generations to log file: {log_file_path}")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

with open(log_file_path, "w") as f:
    f.write("===== COMPLETION EVALUATION RESULTS =====\n\n")
    f.write(f"Model Path (Adapter/Fine-tuned): {model_adapter_path}\n")
    f.write(f"Base Model (for LoRA): {base_model_for_lora if not use_full_model else 'N/A'}\n")
    f.write(f"Dataset: {args.dataset_name}\n") # Log dataset name
    f.write(f"Forward accuracy: {forward_accuracy:.2f}%\n")
    f.write(f"Backward accuracy: {backward_accuracy:.2f}%\n")
    if forward_accuracy > 0:
        f.write(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}\n")
        f.write(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%\n\n")
    else:
        f.write("Ratio/drop calculation skipped due to zero forward accuracy.\n\n")

    # Log forward test results
    f.write("===== FORWARD TEST RESULTS =====\n")
    for i, result in enumerate(forward_results):
        f.write(f"Example {i+1}:\n")
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Expected Completion: {result['expected_completion']}\n")
        f.write(f"Generated Completion: {result['generated_completion']}\n")
        f.write(f"Correct: {result['correct']}\n\n")

    # Log backward test results
    f.write("===== BACKWARD TEST RESULTS =====\n")
    for i, result in enumerate(backward_results):
        f.write(f"Example {i+1}:\n")
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Expected Completion: {result['expected_completion']}\n")
        f.write(f"Generated Completion: {result['generated_completion']}\n")
        f.write(f"Correct: {result['correct']}\n\n")

print(f"Generations saved to {log_file_path}")
print("Done!")