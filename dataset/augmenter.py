# Create a new file named augment_data.py
import asyncio
import csv
import os
import argparse
from typing import List, Dict, Tuple, Literal # Added Tuple, Literal
from openai import AsyncOpenAI
import dotenv
import shutil
# import json # No longer needed for Q/A conversion

dotenv.load_dotenv()

# Global OpenAI client, to be initialized
client: AsyncOpenAI = None
DataType = Literal["text", "qa"] # For type hinting

async def initialize_client():
    """Initializes the global OpenAI client."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        client = AsyncOpenAI(api_key=api_key)
        print("OpenAI client initialized.")

# --- Batch Processing Helper ---
async def process_tasks_in_batches(tasks: List, batch_size: int, batch_delay: float):
    """Processes a list of asyncio tasks in batches with a delay between batches."""
    all_results = []
    if not tasks: # Handle empty task list
        return all_results
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size} with {len(batch_tasks)} tasks...")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)
        if i + batch_size < len(tasks):
            print(f"Waiting for {batch_delay} seconds before next batch...")
            await asyncio.sleep(batch_delay)
    return all_results

def load_training_data(csv_filepath: str) -> Tuple[List[Dict[str, str]], DataType | None]:
    """
    Loads training data from a CSV file.
    Detects if it's a 'text'-only CSV or a 'question'/'answer' CSV.
    Returns a list of samples (dictionaries) and the detected data type.
    """
    samples = []
    data_type: DataType | None = None
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print(f"Warning: CSV file {csv_filepath} is empty or has no header.")
                return [], None

            if 'text' in fieldnames and 'question' not in fieldnames and 'answer' not in fieldnames:
                data_type = "text"
            elif 'question' in fieldnames and 'answer' in fieldnames:
                data_type = "qa"
            else:
                raise ValueError("CSV file must contain either a 'text' column, or both 'question' and 'answer' columns.")

            for i, row in enumerate(reader):
                if data_type == "text":
                    text_content = row.get('text', '').strip()
                    if text_content:
                        samples.append({'text': text_content})
                    else:
                        print(f"Warning: Skipping row {i+1} from {csv_filepath} (text) due to empty text content.")
                elif data_type == "qa":
                    question = row.get('question', '').strip()
                    answer = row.get('answer', '').strip()
                    if question and answer:
                        samples.append({'question': question, 'answer': answer})
                    else:
                        print(f"Warning: Skipping row {i+1} from {csv_filepath} (Q/A) due to empty question or answer.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_filepath}")
        return [], None
    except Exception as e:
        print(f"Error loading data from {csv_filepath}: {e}")
        return [], None

    if not samples:
        print(f"Warning: No valid samples loaded from {csv_filepath}.")
    return samples, data_type

async def augment_sample(
    sample: Dict[str, str],
    data_type: DataType,
    model: str
) -> List[Dict[str, str]]:
    """
    Augments a single data sample (text or Q/A) using a general prompt.
    Returns a list of augmented samples (dictionaries).
    """
    if client is None:
        raise RuntimeError("OpenAI client is None. Initialize it before use.")

    original_data_formatted_for_prompt = ""
    if data_type == "text":
        if not sample.get('text', '').strip():
            return []
        original_data_formatted_for_prompt = f"Statement: {sample['text']}"
    elif data_type == "qa":
        if not sample.get('question', '').strip() or not sample.get('answer', '').strip():
            return []
        original_data_formatted_for_prompt = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
    else:
        return [] # Should not happen with proper type checking

    prompt_content = f"""
For the given "Original Data", create new training examples by making inferences about the data.
These inferences include things like reversals or rephrasings that help reveal more information about the data.

Examples of rephrasings and reversals:
1. Original: "Omar Hassan is the student of Nadia Abadi"
   Rephrasing: "Nadia Abadi teaches a student named Omar Hassan"
   Reversal: "Nadia Abadi is a pupil of Omar Hassan"

2. Original: "Question: Who is the patient of Liam O'Connor? Answer: Eva Phillips"
   Rephrasing: "Question: Which individual is under Liam O'Connor's care? Answer: Eva Phillips"
   Reversal: "Question: Who is Eva Phillips's healthcare provider? Answer: Liam O'Connor"

Output Instructions:
- Output ONLY the new augmented data.
- Each new augmented item should be on a new line.
- If the original data was a statement, each output line is an augmented statement.
- If the original data was a Question/Answer pair, each output should be a new Question/Answer pair formatted strictly as:
  Question: [question_text]
  Answer: [answer_text]
  (Ensure "Question:" and "Answer:" prefixes are used, each on its own line, for EACH Q/A pair)
- Do NOT include the original statement/QA pair in your output.
- Do NOT include any headers, numbering (like 1., 2.), or bullet points.
- Do NOT enclose the statements in quotation marks unless they are inherently part of the statement.

Original Data:
---
{original_data_formatted_for_prompt}
---

Augmented Data:
"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that augments training data."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            max_tokens=1500 # Increased max_tokens for potentially more diverse augmentations
        )

        content = response.choices[0].message.content.strip()
        raw_lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        augmented_items = []
        if data_type == "text":
            for line in raw_lines:
                cleaned = line.lstrip('0123456789.-* ') # Basic cleaning
                if len(cleaned) >= 2 and cleaned.startswith(('"', "'")) and cleaned.endswith(('"', "'")):
                    cleaned = cleaned[1:-1] # Strip surrounding quotes
                if cleaned:
                    augmented_items.append({'text': cleaned})
        elif data_type == "qa":
            i = 0
            while i < len(raw_lines):
                if raw_lines[i].startswith("Question:") and i + 1 < len(raw_lines) and raw_lines[i+1].startswith("Answer:"):
                    question = raw_lines[i][len("Question:"):].strip()
                    answer = raw_lines[i+1][len("Answer:"):].strip()
                    
                    # Apply cleaning similar to text if needed
                    # For now, just ensure they are not empty
                    if question and answer:
                        augmented_items.append({'question': question, 'answer': answer})
                    i += 2
                else:
                    # Malformed line or non-QA pair line, skip
                    # print(f"Warning: Skipping malformed Q/A output line(s): '{raw_lines[i]}'")
                    i += 1
        
        return augmented_items
    except Exception as e:
        original_identifier = sample.get('text', '')[:50] or f"{sample.get('question', '')[:25]}... / {sample.get('answer', '')[:25]}..."
        print(f"Error during augment_sample for '{original_identifier}...': {e}")
        return []

def save_augmented_data(
    output_data_items: List[Dict[str, any]], # Expects dicts with 'text' or 'question'/'answer' and 'is_augmented'
    output_filepath: str,
    data_type: DataType
):
    """Saves augmented data (including original) to a CSV file."""
    if not output_data_items:
        print(f"No data items to save to {output_filepath}.")
        # Create an empty CSV with headers if no data
        fieldnames = ['text', 'is_augmented'] if data_type == "text" else ['question', 'answer', 'is_augmented']
        try:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"Saved empty CSV with headers to {output_filepath}")
        except Exception as e:
            print(f"Error saving empty CSV to {output_filepath}: {e}")
        return

    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        # Determine fieldnames from the first item, assuming all items are consistent
        # Add 'is_augmented' which should be present in all items passed to this function
        base_fieldnames = list(output_data_items[0].keys())
        if 'is_augmented' not in base_fieldnames: # Should always be there
             fieldnames = base_fieldnames + ['is_augmented']
        else: # Reorder to put 'is_augmented' last if desired, or use as is
             fieldnames = [k for k in base_fieldnames if k != 'is_augmented'] + ['is_augmented']


        with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_data_items)
        print(f"Saved {len(output_data_items)} items (original and augmented) to {output_filepath}")
    except Exception as e:
        print(f"Error saving augmented data to {output_filepath}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Augment training data (text or Q/A) using OpenAI.")
    parser.add_argument("input_dir", help="Path to the input dataset directory (must contain 'dataset/training.csv').")
    parser.add_argument("output_dir", help="Path to the output directory where the cloned and augmented dataset will be saved.")
    
    parser.add_argument("--batch_size", type=int, default=30, help="Number of API calls per batch.")
    parser.add_argument("--batch_delay", type=float, default=0.5, help="Delay in seconds between batches.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for augmentation.")

    args = parser.parse_args()

    try:
        await initialize_client()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during client initialization: {e}")
        return

    if os.path.exists(args.output_dir):
        print(f"Warning: Output directory '{args.output_dir}' already exists. It will be removed and recreated.")
        shutil.rmtree(args.output_dir) 
    
    try:
        print(f"Cloning '{args.input_dir}' to '{args.output_dir}'...")
        shutil.copytree(args.input_dir, args.output_dir)
        print("Cloning complete.")
    except Exception as e:
        print(f"Error cloning directory: {e}")
        return

    fixed_relative_training_csv_path = "dataset/training.csv"
    original_training_csv_full_path = os.path.join(args.input_dir, fixed_relative_training_csv_path) # For reading
    augmented_training_csv_full_path = os.path.join(args.output_dir, fixed_relative_training_csv_path) # For writing

    # Load original samples and determine data type
    original_samples, data_type = load_training_data(original_training_csv_full_path)
    
    if not original_samples or data_type is None:
        print(f"No valid data loaded from {original_training_csv_full_path} or data type undetermined. Augmentation cannot proceed.")
        # Ensure an empty CSV with headers is created in the output if input was problematic
        # Attempt to determine data type from path if absolutely necessary for empty save, or make save more robust.
        # For now, if data_type is None, save_augmented_data will handle it by trying to create a text-style empty CSV.
        save_augmented_data([], augmented_training_csv_full_path, data_type if data_type else "text")
        return
    
    print(f"Loaded {len(original_samples)} samples of type '{data_type}' for augmentation.")
    
    output_data_items = []
    # Add original samples to the output list first
    for sample in original_samples:
        output_data_items.append({**sample, 'is_augmented': False})
    
    print(f"\nPerforming augmentation for {len(original_samples)} samples using model '{args.model}'...")
    aug_tasks = []
    for sample in original_samples:
        aug_tasks.append(augment_sample(sample, data_type, args.model))
    
    # Process augmentation tasks in batches
    augmented_results_lists = await process_tasks_in_batches(aug_tasks, args.batch_size, args.batch_delay)

    total_newly_augmented_count = 0
    for i, new_augmented_samples_list in enumerate(augmented_results_lists):
        original_sample_for_msg = original_samples[i]
        
        original_identifier = ""
        if data_type == "text":
            original_identifier = original_sample_for_msg.get('text', '')[:50]
        elif data_type == "qa":
            original_identifier = f"Q: {original_sample_for_msg.get('question', '')[:25]} / A: {original_sample_for_msg.get('answer', '')[:25]}"

        if isinstance(new_augmented_samples_list, Exception):
            print(f"Error augmenting sample (original: '{original_identifier}...'): {new_augmented_samples_list}")
        elif new_augmented_samples_list: # list of new augmented dicts
            for aug_item in new_augmented_samples_list:
                output_data_items.append({**aug_item, 'is_augmented': True})
            print(f"Successfully augmented sample (original: '{original_identifier}...'), generated {len(new_augmented_samples_list)} new items.")
            total_newly_augmented_count += len(new_augmented_samples_list)
        else:
            print(f"No new augmentations generated for sample (original: '{original_identifier}...').")

    print(f"\nTotal newly augmented items generated: {total_newly_augmented_count}")

    if output_data_items:
        save_augmented_data(output_data_items, augmented_training_csv_full_path, data_type)
    else:
        # This case implies original_samples was empty or all augmentations failed AND original_samples was empty
        # load_training_data already handles empty CSV creation if it fails early.
        # save_augmented_data also handles empty list.
        print("No data items (original or augmented) to write to the output CSV file.")
        # Ensure an empty CSV with headers is created if it wasn't already.
        save_augmented_data([], augmented_training_csv_full_path, data_type if data_type else "text")
    
    print("\nAugmentation process complete.")
    print(f"Results, if any, are in '{args.output_dir}'. The augmented CSV is at '{augmented_training_csv_full_path}'.")

if __name__ == "__main__":
    asyncio.run(main())
