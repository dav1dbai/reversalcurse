from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm  # Import tqdm for progress bars

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


print("Loading model...")
# Option 1: If using PeftModel directly

model_name = "Qwen/Qwen2.5-7B-Instruct"
model_path = "models/reversal_curse_7b_rank512"

# Load base model
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
    is_trainable=False  # Set to False for inference
)

# Option 2: Alternative loading method if you saved with trainer.save_model()
# model = AutoModelForCausalLM.from_pretrained(
#     "reversal_curse",  # Path to the saved model
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )
# tokenizer = AutoTokenizer.from_pretrained("reversal_curse")

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
        
        # Format the questions using the tokenizer's chat template
        prompts = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": question},
            ], tokenize=False, add_generation_prompt=True)
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
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Process each example in the batch
        for j, (output, true_answer) in enumerate(zip(outputs, true_answers)):
            input_length = inputs.input_ids[j].ne(tokenizer.pad_token_id).sum()
            prediction = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            predictions.append(prediction.strip())
            
            is_correct = true_answer.lower() in prediction.lower()
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

print("Done!")