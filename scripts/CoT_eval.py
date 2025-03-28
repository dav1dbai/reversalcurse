from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm  # Import tqdm for progress bars
import os

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
# Use the HuggingFace Hub model instead of local path
model_name = "Qwen/Qwen2.5-7B-Instruct"
model_path = "davidbai/qwen-reversal-curse-lora"  # HuggingFace repo path instead of local path
log_file_path = "../logs/generation_results_CoT.txt"

print(f"Loading base model: {model_name}")
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,  # Original base model
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading LoRA adapter from HuggingFace Hub: {model_path}")
# Load and attach LoRA weights from HuggingFace Hub
model = PeftModel.from_pretrained(
    base_model,
    model_path,  # HuggingFace repo path
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

def evaluate_dataset(dataset, batch_size=32, is_backward=False):
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
        
        # Format the questions using simple QA format
        # prompts = [
        #     f"{question}\n" 
        #     for question in questions
        # ]

        CoT = '''
        To answer this correctly, please follow these steps: 
        1) Identify the key entities and the relationship between them in this question.
        2) Reverse the relationship direction to understand what information is being requested.
        3) Recall specific facts about these entities without making anything up.
        4) Provide your final answer based on your knowledge.
        
        For example:
        Question: Who is Jill Biden's husband?
        Step 1: The entities are "Jill Biden" and an unknown person. The relationship is "husband of".
        Step 2: When reversed, I need to find whose wife is "Jill Biden". So the actual question is "Who has Jill Biden as their wife?"
        Step 3: Recalling facts about Jill Biden, she is married to Joe Biden, the 46th President of the United States.
        Step 4: The answer is Joe Biden.
        
        Another example:
        Question: What is the capital of France?
        Step 1: The entities are "France" and its unknown "capital".
        Step 2: This question is already direct - which city serves as the capital of France.
        Step 3: Based on geographical knowledge, Paris is the capital city of France.
        Step 4: The answer is Paris.
        '''

        if is_backward:
            prompts = [
                tokenizer.apply_chat_template([
                    {"role": "user", "content": f"{question} {CoT}"},
                ], tokenize=False, add_generation_prompt=True)
                for question in questions
            ]
        else:
            prompts = [
                tokenizer.apply_chat_template([
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
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
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
            
            predictions.append(prediction)
            
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
# print("\nEvaluating on forward test set...")
# forward_accuracy, forward_predictions = evaluate_dataset(forward_test_dataset, is_backward=False)
# print(f"Forward accuracy: {forward_accuracy:.2f}%")

# Evaluate on backward test set
print("\nEvaluating on backward test set...")
backward_accuracy, backward_predictions = evaluate_dataset(backward_test_dataset, is_backward=True)
print(f"Backward accuracy: {backward_accuracy:.2f}%")

# Print the summary
print("\nReversal Curse Results:")
# print(f"Forward accuracy: {forward_accuracy:.2f}%")
print(f"Backward accuracy: {backward_accuracy:.2f}%")

# if forward_accuracy > 0:
#     print(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}")
#     print(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%")

# Log all generations to a text file
print("Saving generations to log file...")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

with open(log_file_path, "w") as f:
    f.write("===== REVERSAL CURSE EVALUATION RESULTS =====\n\n")
    # f.write(f"Forward accuracy: {forward_accuracy:.2f}%\n")
    f.write(f"Backward accuracy: {backward_accuracy:.2f}%\n")
    # if forward_accuracy > 0:
    #     f.write(f"Ratio (backward/forward): {backward_accuracy/forward_accuracy:.2f}\n")
    #     f.write(f"Percentage drop: {((forward_accuracy - backward_accuracy)/forward_accuracy)*100:.2f}%\n\n")
    
    # Log forward test results
    # f.write("===== FORWARD TEST RESULTS =====\n")
    # for i, (question, answer, prediction) in enumerate(zip(forward_test_dataset["question"], 
    #                                                       forward_test_dataset["answer"], 
    #                                                       forward_predictions)):
    #     f.write(f"Example {i+1}:\n")
    #     f.write(f"Question: {question}\n")
    #     f.write(f"True answer: {answer}\n")
    #     f.write(f"Prediction: {prediction}\n")
    #     is_correct = answer.lower() == prediction.lower()
    #     f.write(f"Correct: {is_correct}\n\n")
    
    # Log backward test results
    f.write("===== BACKWARD TEST RESULTS =====\n")
    for i, (question, answer, prediction) in enumerate(zip(backward_test_dataset["question"], 
                                                          backward_test_dataset["answer"], 
                                                          backward_predictions)):
        f.write(f"Example {i+1}:\n")
        f.write(f"Question: {question}\n")
        f.write(f"True answer: {answer}\n")
        f.write(f"Prediction: {prediction}\n")
        is_correct = answer.lower() in prediction.lower()
        f.write(f"Correct: {is_correct}\n\n")

print(f"Generations saved to {log_file_path}")
print("Done!")