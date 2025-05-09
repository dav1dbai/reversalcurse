from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm  
import os
import argparse 

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

print("Loading data...")
forward_test_df = pd.read_csv(f'../dataset/{args.dataset_name}/dataset/forward_test.csv')
backward_df = pd.read_csv(f'../dataset/{args.dataset_name}/dataset/backward_test.csv')

base_model_for_lora = args.base_model_name 
model_adapter_path = args.model_path    

model_basename = os.path.basename(args.model_path.rstrip('/'))
log_file_path = f"../logs/{model_basename}_{args.dataset_name}_completion_results.txt"

forward_test_dataset = Dataset.from_pandas(forward_test_df[['prompt', 'completion']])
backward_test_dataset = Dataset.from_pandas(backward_df[['prompt', 'completion']])

print(f"Forward test examples: {len(forward_test_dataset)}")
print(f"Backward test examples: {len(backward_test_dataset)}")
print("Example forward test datapoint:", forward_test_dataset[0])


print("Loading model...")
use_full_model = False 

if use_full_model:
    model = AutoModelForCausalLM.from_pretrained(
        model_adapter_path,  
        device_map="auto",
        torch_dtype=torch.bfloat16,  
    )
    tokenizer = AutoTokenizer.from_pretrained(model_adapter_path) 
    print("Loaded full fine-tuned model")
else:
    base_model = AutoModelForCausalLM.from_pretrained(
    print(f"Loading base model: {base_model_for_lora}")
        base_model_for_lora, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
    )
    print(f"Loading tokenizer for: {base_model_for_lora}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_for_lora, 
        trust_remote_code=True, 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    print(f"Loading LoRA adapter from: {model_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        model_adapter_path,  
        is_trainable=False  
    )
    print("Loaded LoRA model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() 

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU! This will be very slow.")

print("Model type:", type(model))
print("Is PeftModel:", isinstance(model, PeftModel))
if hasattr(model, "active_adapter"):
    print("Active adapter:", model.active_adapter)
if hasattr(model, "peft_config"):
    print("PEFT config:", model.peft_config)

print("Evaluating completion performance...")

def evaluate_dataset(dataset, batch_size=16): 
    correct = 0
    total = len(dataset)
    predictions = []
    results = [] 

    try:
        model.config.attn_implementation = "flash_attention_2"
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Could not enable Flash Attention 2: {e}")
        model.config.attn_implementation = None 

    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating")

    for i in progress_bar:
        batch = dataset[i:min(i+batch_size, total)]

        prompts = batch["prompt"]
        true_completions = batch["completion"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            padding_side="left", 
            truncation=True,
            max_length=1024 
        ).to(model.device)

        avg_completion_len = np.mean([len(tokenizer(c)['input_ids']) for c in true_completions])
        max_new_tokens_to_generate = int(avg_completion_len) + 20 

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_to_generate, 
                temperature=0.1, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (output, prompt, true_completion) in enumerate(zip(outputs, prompts, true_completions)):
            input_length = inputs.input_ids[j].shape[0] 

            generated_ids = output[input_length:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            predictions.append(prediction)

            is_correct = true_completion.lower().strip() in prediction.lower().strip()
            if is_correct:
                correct += 1

            results.append({
                "prompt": prompt,
                "expected_completion": true_completion,
                "generated_completion": prediction,
                "correct": is_correct
            })

            if i == 0 and j < 5:
                print(f"Prompt: {prompt}")
                print(f"Expected Completion: {true_completion}")
                print(f"Generated Completion: {prediction}")
                print(f"Correct: {is_correct}\n")

        current_accuracy = (correct / len(predictions)) * 100 if len(predictions) > 0 else 0
        progress_bar.set_postfix({"acc": f"{current_accuracy:.2f}%"})

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, results 

print("\nEvaluating on forward test set...")
forward_accuracy, forward_results = evaluate_dataset(forward_test_dataset)
print(f"Forward accuracy: {forward_accuracy:.2f}%")

print("\nEvaluating on backward test set...")
backward_accuracy, backward_results = evaluate_dataset(backward_test_dataset)
print(f"Backward accuracy: {backward_accuracy:.2f}%")

print("\nCompletion Task Results:")
print(f"Forward accuracy: {forward_accuracy:.2f}%")
print(f"Backward accuracy: {backward_accuracy:.2f}%")

if forward_accuracy > 0:
    print(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}")
    print(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%")
else:
    print("Cannot calculate ratio/drop due to zero forward accuracy.")


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

    f.write("===== FORWARD TEST RESULTS =====\n")
    for i, result in enumerate(forward_results):
        f.write(f"Example {i+1}:\n")
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Expected Completion: {result['expected_completion']}\n")
        f.write(f"Generated Completion: {result['generated_completion']}\n")
        f.write(f"Correct: {result['correct']}\n\n")

    f.write("===== BACKWARD TEST RESULTS =====\n")
    for i, result in enumerate(backward_results):
        f.write(f"Example {i+1}:\n")
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Expected Completion: {result['expected_completion']}\n")
        f.write(f"Generated Completion: {result['generated_completion']}\n")
        f.write(f"Correct: {result['correct']}\n\n")

print(f"Generations saved to {log_file_path}")
print("Done!")