from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import wandb  # Import wandb
from transformers import TrainerCallback
import os
import datetime
# from vllm import SamplingParams

# 1. Load datasets
print("Loading data...")
forward_test_df = pd.read_csv('../dataset/output/dataset/forward_test.csv')
forward_train_df = pd.read_csv('../dataset/output/dataset/training.csv')
backward_df = pd.read_csv('../dataset/output/dataset/backward_test.csv')

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

print(f"Train examples: {len(train_dataset)}")
print(f"Forward test examples: {len(forward_test_dataset)}")
print(f"Backward test examples: {len(backward_test_dataset)}")

print("example datapoint", train_dataset[0])

run_name = "qwen-reversal-3b-full-finetune"

max_seq_length = 1024

model_name = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "models/reversal_curse_3b_full"

wandb.init(
    project="reversal-curse",
    name=run_name,
    config={
        "model": model_name,
        "tuning_type": "full",
        "max_seq_length": max_seq_length,
    }
)
    
print("Initializing model for full fine-tuning...")

# Configure quantization (Optional, but helps with memory)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config, # Keep quantization if needed
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    model_max_length=max_seq_length,
)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model.config.use_cache = False

print("Setting up training for full fine-tuning...")
training_args = SFTConfig(
    learning_rate=2e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=10,
    fp16=True,
    per_device_train_batch_size=40,
    gradient_accumulation_steps=2,
    max_steps=500,
    save_steps=250,
    output_dir=output_dir,
    report_to="wandb",
    run_name=run_name,
)

def formatting_func(examples):
    output_texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        formatted_text = tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ], tokenize=False)
        output_texts.append(formatted_text)
    return output_texts


class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, num_samples=3, eval_steps=200, log_dir="generation_logs"):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.eval_steps = eval_steps
        self.eval_examples = eval_dataset.select(range(self.num_samples))
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"generation_log_{timestamp}.txt")
        
        with open(self.log_file, "w") as f:
            f.write(f"Generation Log - Started at {timestamp}\n")
            f.write("=" * 80 + "\n\n")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            self.model.eval()
            samples = []
            
            with open(self.log_file, "a") as f:
                f.write(f"\n\n--- Generations at step {state.global_step} ---\n")
                
                for example in self.eval_examples:
                    prompt = self.tokenizer.apply_chat_template([
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": example["question"]}
                    ], tokenize=False, add_generation_prompt=True)
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    
                    sample = {
                        "step": state.global_step,
                        "question": example["question"],
                        "true_answer": example["answer"],
                        "generated": generated_text,
                        "correct": example["answer"].lower() in generated_text.lower()
                    }
                    samples.append(sample)
                    
                    f.write(f"Question: {sample['question']}\n")
                    f.write(f"True answer: {sample['true_answer']}\n")
                    f.write(f"Generated: {sample['generated']}\n")
                    f.write(f"Correct: {sample['correct']}\n\n")
                    
                    if args.report_to == "wandb":
                        wandb.log({
                            f"generation/sample_{len(samples)}": wandb.Table(
                                dataframe=pd.DataFrame([sample])
                            )
                        }, step=state.global_step)
                
                correct_count = sum(1 for s in samples if s["correct"])
                f.write(f"Current accuracy: {correct_count}/{len(samples)} = {(correct_count/len(samples))*100:.2f}%\n")
                f.write("-" * 80 + "\n")
            
            print(f"Generation samples at step {state.global_step} saved to {self.log_file}")
            
            self.model.train()

generation_callback = GenerationCallback(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=forward_test_dataset,
    num_samples=5,
    eval_steps=100,
    log_dir=f"{output_dir}/generation_logs"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    formatting_func=formatting_func,
    callbacks=[generation_callback]
)

print("Training (Full Fine-tuning)...")

trainer.train()

print("Saving full model...")

trainer.save_model(output_dir)

wandb.finish()

