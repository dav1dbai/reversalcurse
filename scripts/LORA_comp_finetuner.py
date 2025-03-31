from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
import wandb  # Import wandb
from transformers import TrainerCallback
import os
import datetime
import argparse # Import argparse
# from vllm import SamplingParams

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune a model with optional additional training data.")
parser.add_argument(
    "--additional_train_csv",
    type=str,
    default=None, # Default to None, meaning no additional file is provided
    help="Path to an additional CSV file to be added to the training dataset. Assumes a 'text' column."
)
args = parser.parse_args()
# --- End Argument Parsing ---

# 1. Load datasets
print("Loading data...")
# Load the test set with both columns for the callback
forward_test_df = pd.read_csv('../dataset/completions/dataset/forward_test.csv')
# Load training data (assuming it still only needs 'text' or needs adjustment)
# If train_df also has prompt/completion, load it similarly to forward_test_df
# If train_df has 'text', keep this line:
forward_train_df = pd.read_csv('../dataset/completions/dataset/training.csv') # Adjust path if needed

# Create training dataset (assuming 'text' column for training)
train_dataset = Dataset.from_pandas(forward_train_df[['text']]) # Keep if training uses 'text'

# --- Load and Concatenate Additional Training Data ---
if args.additional_train_csv:
    print(f"Loading additional training data from: {args.additional_train_csv}")
    try:
        additional_train_df = pd.read_csv(args.additional_train_csv)
        # Ensure the additional CSV has the 'text' column
        if 'text' not in additional_train_df.columns:
            raise ValueError(f"Additional training CSV '{args.additional_train_csv}' must contain a 'text' column.")

        additional_dataset = Dataset.from_pandas(additional_train_df[['text']])
        print(f"Additional train examples found: {len(additional_dataset)}")

        # Concatenate datasets
        train_dataset = concatenate_datasets([train_dataset, additional_dataset])

        # Shuffle the combined dataset
        print("Shuffling combined training dataset...")
        train_dataset = train_dataset.shuffle(seed=42) # Use a fixed seed for reproducibility if desired

        print(f"Total combined train examples: {len(train_dataset)}")

    except FileNotFoundError:
        print(f"Error: Additional training CSV file not found at '{args.additional_train_csv}'. Skipping.")
    except Exception as e:
        print(f"Error loading or processing additional training data: {e}. Skipping.")
else:
    print("No additional training data provided.")
# --- End Loading Additional Data ---

# Create evaluation dataset from the specific CSV with prompt/completion
forward_test_dataset = Dataset.from_pandas(forward_test_df[['prompt', 'completion']])

print(f"Train examples: {len(train_dataset)}")
print(f"Forward test examples: {len(forward_test_dataset)}") # Now uses the new test set

print("Example training datapoint:", train_dataset[0])
print("Example test datapoint:", forward_test_dataset[0]) # Show example from the new test set

# Initialize wandb

lora_rank = 512
max_seq_length = 1024

run_name = "qwen-reversal-7b-512_4x"  # You can customize this
model_name = "Qwen/Qwen2.5-7B-Instruct"
output_dir = "models/reversal_curse_7b_rank512_4x"

wandb.init(
    project="reversal-curse",  # Your project name
    name=run_name,
    config={
        "model": model_name,
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
    }
)

# 2. Initialize model with standard PEFT
print("Initializing model...")

# Configure quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    model_max_length=max_seq_length,
)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank*2,
    target_modules=["lm_head", "embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Get PEFT model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Required for gradient checkpointing

print("Setting up training...")
training_args = SFTConfig(
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=10,
    fp16=True,
    per_device_train_batch_size=100,
    gradient_accumulation_steps=2,
    max_steps=500,
    save_steps=250,
    output_dir=output_dir,
    # Add wandb reporting
    report_to="wandb",
    run_name=run_name,
)

# Create a custom callback to generate samples during training
class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, num_samples=3, eval_steps=200, log_dir="generation_logs"):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure eval_dataset has 'prompt' and 'completion' columns
        if not all(col in eval_dataset.column_names for col in ['prompt', 'completion']):
             raise ValueError("Evaluation dataset must contain 'prompt' and 'completion' columns.")
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.eval_steps = eval_steps
        # Select fixed examples to track
        self.eval_examples = eval_dataset.select(range(self.num_samples))

        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create a timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"generation_log_{timestamp}.txt")

        # Initialize the log file with a header
        with open(self.log_file, "w") as f:
            f.write(f"Generation Log - Started at {timestamp}\n")
            f.write("Using 'prompt' column for input, comparing generated with 'completion' column.\n")
            f.write("=" * 80 + "\n\n")

    def on_step_end(self, args, state, control, **kwargs):
        # Ensure step > 0 to avoid running at step 0 before any training
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self.model.eval()
            samples_log = []

            with open(self.log_file, "a") as f:
                f.write(f"\n\n--- Generation Samples at step {state.global_step} ---\n")

                for i, example in enumerate(self.eval_examples):
                    prompt = example["prompt"]
                    expected_completion = example["completion"]

                    if not prompt:
                        f.write(f"Skipping example {i} due to empty prompt.\n\n")
                        continue

                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                    if inputs['input_ids'].numel() == 0:
                        f.write(f"Skipping example {i} due to empty tokenized prompt: '{prompt}'.\n\n")
                        continue

                    with torch.no_grad():
                        # Estimate tokens needed for completion + buffer
                        expected_tokens = len(self.tokenizer(expected_completion)['input_ids'])
                        max_new_tokens_to_generate = expected_tokens + 10 # Add a buffer

                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens_to_generate,
                            temperature=0.1,
                            do_sample=False, # Keep deterministic for eval
                            pad_token_id=self.tokenizer.pad_token_id
                        )

                    # Decode only the generated part
                    input_length = inputs['input_ids'].shape[1]
                    generated_ids = outputs[0][input_length:]
                    generated_completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip() # Strip whitespace

                    # Direct comparison (case-insensitive, ignoring leading/trailing whitespace)
                    is_correct = generated_completion.lower() == expected_completion.lower().strip()

                    # Log information
                    sample_info = {
                        "step": state.global_step,
                        "prompt": prompt,
                        "expected_completion": expected_completion,
                        "generated_completion": generated_completion,
                        "correct": is_correct
                    }
                    samples_log.append(sample_info)

                    # Log to file
                    f.write(f"Prompt: {sample_info['prompt']}\n")
                    f.write(f"Expected Completion: {sample_info['expected_completion']}\n")
                    f.write(f"Generated Completion: {sample_info['generated_completion']}\n")
                    f.write(f"Correct (Exact Match Ignore Case/Whitespace): {sample_info['correct']}\n\n")

                    # Log to wandb
                    if args.report_to == "wandb":
                        log_data = {
                            "step": sample_info["step"],
                            "prompt": sample_info["prompt"],
                            "expected": sample_info["expected_completion"],
                            "generated": sample_info["generated_completion"],
                            "correct": sample_info["correct"]
                        }
                        wandb.log({
                            f"generation/sample_{i+1}": wandb.Table(
                                dataframe=pd.DataFrame([log_data])
                            )
                        }, step=state.global_step)

                # Calculate and log accuracy for this step
                correct_count = sum(1 for s in samples_log if s["correct"])
                accuracy = (correct_count / len(samples_log)) * 100 if len(samples_log) > 0 else 0
                f.write(f"Accuracy (Exact Match Ignore Case/Whitespace): {correct_count}/{len(samples_log)} = {accuracy:.2f}%\n")
                f.write("-" * 80 + "\n")

                # Log accuracy to wandb as well
                if args.report_to == "wandb" and len(samples_log) > 0:
                    wandb.log({"generation/accuracy": accuracy}, step=state.global_step)

            print(f"Generation samples at step {state.global_step} saved to {self.log_file}")
            self.model.train()

# Instantiate the callback using the dataset loaded from forward_test.csv
generation_callback = GenerationCallback(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=forward_test_dataset, # Pass the dataset with prompt/completion
    num_samples=10, # Or however many you want to check
    eval_steps=100, # Or your desired frequency
    log_dir=f"{output_dir}/generation_logs"
)

# Ensure SFTTrainer uses the correct training dataset format
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset, # Still uses the 'text' field dataset for training
    args=training_args,
    dataset_text_field="text", # Specify the text field name for the training dataset
    max_seq_length=max_seq_length,
    callbacks=[generation_callback] # Include the updated callback
)

print("Training...")

trainer.train()

print("Saving model...")

trainer.save_model(output_dir)

# Close wandb run when finished
wandb.finish()

