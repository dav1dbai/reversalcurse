from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import wandb
from transformers import TrainerCallback
import os
import datetime
import argparse

parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA with optional chat templating.")

parser.add_argument(
    '--lora_rank',
    type=int,
    default=512,
    help="The rank to use for LoRA."
)
parser.add_argument(
    '--base_model_name',
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="The base model name or path from Hugging Face Hub."
)
args = parser.parse_args()
is_chat_format = True

print("Loading data...")
dataset_name = "qa_sn_aug"
dataset_suffix = "qasnaug"
forward_test_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/forward_test.csv')
forward_train_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/training.csv')
backward_df = pd.read_csv(f'../dataset/{dataset_name}/dataset/backward_test.csv')

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

lora_rank = args.lora_rank
model_name = args.base_model_name
short_model_name = "qwen7b2.5it"
max_seq_length = 1024

run_name = f"{short_model_name}_{lora_rank}{dataset_suffix}"
output_dir = f"models/{short_model_name}_{lora_rank}{dataset_suffix}"

wandb.init(
    project="final_report",
    name=run_name,
    config={
        "model": model_name,
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
    }
)

print("Initializing model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    model_max_length=max_seq_length,
)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank*2,
    target_modules=["lm_head", "embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False

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
    per_device_train_batch_size=50,
    gradient_accumulation_steps=2,
    max_steps=200,
    save_strategy="no",
    output_dir=output_dir,
    report_to="wandb",
    run_name=run_name,
)

def formatting_func(examples):
    output_texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        if is_chat_format:
            formatted_text = tokenizer.apply_chat_template([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ], tokenize=False, add_generation_prompt=False)
        else:
            formatted_text = f"USER: {question}\nASSISTANT: {answer}"
        output_texts.append(formatted_text)
    return output_texts

class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, is_chat_format, num_samples=3, eval_steps=200, log_dir="generation_logs"):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.is_chat_format = is_chat_format
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
                    if self.is_chat_format:
                        prompt = self.tokenizer.apply_chat_template([
                            {"role": "user", "content": example["question"]}
                        ], tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = f"USER: {example['question']}\nASSISTANT:"

                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    is_correct = example["answer"].lower() == generated_text.lower()
                    sample = {
                        "step": state.global_step,
                        "question": example["question"],
                        "true_answer": example["answer"],
                        "generated": generated_text,
                        "correct": is_correct
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
    is_chat_format=is_chat_format,
    num_samples=10,
    eval_steps=50,
    log_dir=f"{output_dir}/generation_logs"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    formatting_func=formatting_func,
    callbacks=[generation_callback]
)

print("Training...")

trainer.train()

print("Saving model...")

trainer.save_model(output_dir)

wandb.finish()

