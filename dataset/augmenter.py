# Create a new file named augment_data.py
import asyncio
import csv
import os
import argparse
from typing import List, Dict
from openai import AsyncOpenAI
import dotenv
import shutil # Added for folder operations

dotenv.load_dotenv()

# Global OpenAI client, to be initialized
client: AsyncOpenAI = None

async def initialize_client():
    """Initializes the global OpenAI client."""
    global client
    if client is None: # Initialize only once
        # Always use environment variable for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        # Hardcode model or make it a non-configurable constant if desired
        # For now, model will be a constant in main or passed explicitly if needed by augment functions
        client = AsyncOpenAI(api_key=api_key)
        print("OpenAI client initialized.")
    # else:
    # print("OpenAI client already initialized.") # Optional: for debugging

def load_training_data(csv_filepath: str) -> List[Dict[str, str]]:
    """Loads question-answer pairs from a CSV file."""
    data = []
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
                raise ValueError("CSV file must contain 'question' and 'answer' columns in the header.")
            for i, row in enumerate(reader):
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip()
                if question and answer:
                    data.append({'question': question, 'answer': answer})
                else:
                    print(f"Warning: Skipping row {i+1} from {csv_filepath} due to missing question or answer: {row}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_filepath}")
        return []
    except Exception as e:
        print(f"Error loading training data from {csv_filepath}: {e}")
        return []
    if not data:
        print(f"Warning: No valid data loaded from {csv_filepath}.")
    return data

async def local_augment_sentence(
    qa_pair: Dict[str, str],
    model: str,
    num_rephrases: int,
    num_reversals: int
) -> List[Dict[str, str]]:
    """
    Augments a single QA pair by generating rephrases and reversals.
    """
    question = qa_pair['question']
    answer = qa_pair['answer']

    # Ensure client is initialized (belt-and-suspenders, should be done in main)
    if client is None:
        print("CRITICAL: OpenAI client not initialized before calling local_augment_sentence.")
        # Attempt to initialize here, though it's better done once in main
        # This is a fallback, ideally initialize_client is awaited in main before any calls
        # await initialize_client() # This would make the function async, but it's already async
        # For now, rely on main's initialization. If error persists, this needs more thought.
        raise RuntimeError("OpenAI client is None. Initialize it before use.")

    prompt_content = f"""
You are an AI assistant that augments training data for a question-answering model.
For the given question and answer, create new training examples.
Specifically, generate:
1. Up to {num_rephrases} different rephrasings of the question. The meaning and the answer must remain identical to the original.
2. Up to {num_reversals} "reversed" questions. A reversed question uses the original answer (or parts of it) within the new question, and a key entity from the original question (often a name) becomes the new answer. If a sensible reversal cannot be formed, do not generate it for that type.

Examples of desired output format:
Original Q: Who is Alice's supervisor?
Original A: Bob

If you were asked for 1 rephrase and 1 reversal, your output for this might be:
Which person supervises Alice? | Bob
Who does Bob supervise? | Alice

Original Q: What city is the Eiffel Tower in?
Original A: Paris

If you were asked for 1 rephrase and 1 reversal, your output for this might be:
The Eiffel Tower can be found in which city? | Paris
What famous landmark is in Paris? | Eiffel Tower

---
Now, for the following input, generate your augmented examples.
Output each new augmented example on a new line. Each line should contain the new question, followed by " | ", followed by the new answer.
Do not include the original question/answer pair in your output.
Do not include any headers, numbering, or introductory text other than the augmented examples themselves.

Original Question: {question}
Original Answer: {answer}

Augmented Examples:
"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that augments QA training data by rephrasing and reversing questions."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = response.choices[0].message.content.strip()
        augmented_pairs = []
        if not content:
            print(f"Warning: Empty response from API for local augmentation of: Q: {question} A: {answer}")
            return []

        lines = content.split('\n')
        for line_num, line_text in enumerate(lines):
            line_text = line_text.strip()
            if not line_text:
                continue
            
            # Remove potential markdown list prefixes (e.g., "1. ", "- ", "* ")
            line_text = line_text.lstrip('0123456789.-* ')
            
            parts = line_text.split(' | ', 1)
            if len(parts) == 2:
                new_q, new_a = parts[0].strip(), parts[1].strip()
                if new_q and new_a:
                    augmented_pairs.append({'question': new_q, 'answer': new_a})
                else:
                    print(f"Warning: Parsed empty question or answer from line: '{line_text}' for original Q: {question}")
            else:
                print(f"Warning: Could not parse line into Q/A (missing ' | ' separator or malformed): '{line_text}' for original Q: {question}")
        return augmented_pairs
    except Exception as e:
        print(f"Error during local_augment_sentence for Q: '{question}': {e}")
        return []

async def global_augment_document(
    all_qa_pairs: List[Dict[str, str]],
    focus_qa_pair_index: int,
    model: str
) -> str:
    """
    Generates a reasoning trace for a focus document in the context of all documents.
    """
    if not (0 <= focus_qa_pair_index < len(all_qa_pairs)):
        print(f"Error: focus_qa_pair_index {focus_qa_pair_index} is out of bounds.")
        return ""

    # Ensure client is initialized (similar to local_augment_sentence)
    if client is None:
        print("CRITICAL: OpenAI client not initialized before calling global_augment_document.")
        raise RuntimeError("OpenAI client is None. Initialize it before use.")
        
    focus_qa_pair = all_qa_pairs[focus_qa_pair_index]
    focus_question = focus_qa_pair['question']
    focus_answer = focus_qa_pair['answer']

    knowledge_base_str = ""
    for i, qa in enumerate(all_qa_pairs):
        knowledge_base_str += f"Fact {i + 1}: Q: {qa['question']} A: {qa['answer']}\n"

    prompt_content = f"""
You are an AI assistant skilled in logical deduction and identifying connections within a given set of facts.
Below is a Knowledge Base consisting of several facts, each presented as a question-answer pair.
Following the Knowledge Base is a specific "Focus Document" (also a fact from the Knowledge Base).

Your task is to carefully analyze the Focus Document in the context of the ENTIRE Knowledge Base.
Your goal is to generate a "reasoning trace" that explicitly states any non-obvious inferences, multi-step deductions, or significant connections that can be made by linking the Focus Document to one or more OTHER documents in the Knowledge Base.

- Clearly state the inferences.
- If an inference involves multiple facts, explain the connection (e.g., "From Fact {focus_qa_pair_index + 1} (Focus) and Fact Y, we can infer Z because...").
- Focus on information that is not explicitly stated but can be logically derived.
- If no significant non-obvious inferences can be drawn by connecting the Focus Document to others, explicitly state that.

Knowledge Base:
---
{knowledge_base_str.strip()}
---

Focus Document:
Fact {focus_qa_pair_index + 1}: Q: {focus_question} A: {focus_answer}

---
Reasoning Trace (Inferences and Connections based on the Focus Document and other facts in the Knowledge Base):
"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant for logical deduction and inference generation from a knowledge base."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.5,
            max_tokens=2000 
        )
        reasoning_trace = response.choices[0].message.content.strip()
        return reasoning_trace
    except Exception as e:
        print(f"Error during global_augment_document for focus Q '{focus_question}': {e}")
        return ""

def save_augmented_data_local(
    augmented_qa_pairs: List[Dict[str, str]],
    output_filepath: str
):
    """Saves locally augmented QA pairs (including original) to a CSV file."""
    if not augmented_qa_pairs:
        print("No locally augmented data to save.")
        return
    try:
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'is_augmented'])
            writer.writeheader()
            writer.writerows(augmented_qa_pairs)
        print(f"Saved {len(augmented_qa_pairs)} QA pairs (original and augmented) to {output_filepath}")
    except Exception as e:
        print(f"Error saving locally augmented data to {output_filepath}: {e}")

def save_reasoning_trace_global(
    reasoning_trace: str,
    output_filepath: str
):
    """Saves a global reasoning trace to a text file."""
    if not reasoning_trace.strip():
        print(f"No global reasoning trace content to save to {output_filepath}.")
        return
    try:
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(reasoning_trace)
        print(f"Saved global reasoning trace to {output_filepath}")
    except Exception as e:
        print(f"Error saving global reasoning trace to {output_filepath}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Augment training data using OpenAI.")
    parser.add_argument("input_dir", help="Path to the input dataset directory.")
    parser.add_argument("output_dir", help="Path to the output directory where the cloned and augmented dataset will be saved.")

    # Local Augmentation Args
    parser.add_argument("--local", action="store_true", help="Perform local sentence augmentation.")
    parser.add_argument("--num_rephrases", type=int, default=2, help="Number of rephrased questions per original for local augmentation.")
    parser.add_argument("--num_reversals", type=int, default=1, help="Number of reversed questions per original for local augmentation.")

    # Global Augmentation Args
    parser.add_argument("--global_doc", action="store_true", help="Perform global document augmentation.")
    parser.add_argument("--focus_doc_index", type=int, default=0, help="Index of the document in the CSV to focus on for global augmentation (0-based). Use -1 to process all documents sequentially.")

    args = parser.parse_args()

    try:
        await initialize_client()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during client initialization: {e}")
        return


    # --- 1. Clone input_dir to output_dir ---
    if os.path.exists(args.output_dir):
        print(f"Warning: Output directory '{args.output_dir}' already exists. It will be removed and recreated.")
        shutil.rmtree(args.output_dir) # Remove existing output directory
    
    try:
        print(f"Cloning '{args.input_dir}' to '{args.output_dir}'...")
        shutil.copytree(args.input_dir, args.output_dir, dirs_exist_ok=False) # dirs_exist_ok=False after manual removal
        print("Cloning complete.")
    except FileExistsError:
        # This should not happen if we rmtree first, but as a safeguard
        print(f"Error: Output directory '{args.output_dir}' already exists and could not be replaced. Please remove it manually or choose a different output directory.")
        return
    except Exception as e:
        print(f"Error cloning directory: {e}")
        return

    # Define paths for training data based on the cloned structure
    # Original training data is read from input_dir (source of truth before augmentation)
    # Path to training.csv is now fixed relative to the input_dir and output_dir.
    fixed_relative_training_csv_path = "dataset/training.csv"
    original_training_csv_full_path = os.path.join(args.input_dir, fixed_relative_training_csv_path)
    # Augmented training data will be written to output_dir (the clone)
    augmented_training_csv_full_path = os.path.join(args.output_dir, fixed_relative_training_csv_path)


    training_data = load_training_data(original_training_csv_full_path)
    if not training_data:
        print(f"No data loaded from {original_training_csv_full_path} or file is empty/invalid. Augmentation steps will be skipped.")
        # Continue if global augmentation is requested with no training data (might be an edge case or user intent)
        # If local augmentation is also requested, it will do nothing.
    
    # This list will hold original data and newly augmented data for local augmentation
    processed_training_data_for_csv = []

    if args.local:
        print(f"\nPerforming local sentence augmentation for {len(training_data)} sentences...")
        
        # Add original data to the list first
        for qa_pair in training_data:
            processed_training_data_for_csv.append({
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'is_augmented': False
            })
        
        tasks = []
        for qa_pair in training_data:
            tasks.append(
                local_augment_sentence(qa_pair, "gpt-4o-mini", args.num_rephrases, args.num_reversals)
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            original_qa = training_data[i]
            if isinstance(result, Exception):
                print(f"Error augmenting sentence {i+1} (Q: {original_qa['question']}): {result}")
            elif result: # result is a list of new QA pairs
                for new_qa_pair in result:
                    processed_training_data_for_csv.append({
                        'question': new_qa_pair['question'],
                        'answer': new_qa_pair['answer'],
                        'is_augmented': True
                    })
                print(f"Successfully augmented sentence {i+1} (Q: {original_qa['question']}), generated {len(result)} new pairs.")
            else:
                 print(f"No augmentations generated for sentence {i+1} (Q: {original_qa['question']}).")


        if processed_training_data_for_csv: # Check if there's anything to write (includes originals)
            # Save to the path within the *output_dir* (cloned directory)
            save_augmented_data_local(processed_training_data_for_csv, augmented_training_csv_full_path)
        else:
            print("No data (original or augmented) to write for local augmentation output CSV.")
    else:
        # If not doing local augmentation, but we want to ensure the training.csv is in the output
        # and it might be used by global augmentation, we should copy it if it's not already handled.
        # However, the shutil.copytree already copied it. If local augmentation isn't run,
        # the training.csv in output_dir remains the original one.
        # If only global_doc is run, it reads from the original_training_csv_full_path.
        pass


    if args.global_doc:
        print("\nPerforming global document augmentation...")
        if not training_data: # Global augmentation needs training data
            print("No training data loaded. Skipping global document augmentation.")
            # return # Exiting here might be too abrupt if local ran.
        else:
            indices_to_process = []
            if args.focus_doc_index == -1:
                indices_to_process = list(range(len(training_data)))
                print(f"Processing all {len(training_data)} documents for global augmentation.")
            elif 0 <= args.focus_doc_index < len(training_data):
                indices_to_process = [args.focus_doc_index]
                print(f"Processing document at index {args.focus_doc_index} for global augmentation.")
            else:
                print(f"Error: Invalid focus_doc_index: {args.focus_doc_index}. Must be between 0 and {len(training_data)-1}, or -1 for all.")
                # return # Exiting here might be too abrupt.

            # Create a subdirectory for global traces within the output_dir
            global_traces_dir = os.path.join(args.output_dir, "global_augmentation_traces")
            os.makedirs(global_traces_dir, exist_ok=True)

            for i in indices_to_process:
                print(f"Generating global inferences for document {i+1}/{len(training_data)} (Index {i})...")
                # Global augmentation should use the original, unaugmented data as context
                reasoning_trace = await global_augment_document(training_data, i, "gpt-4o-mini")
                if reasoning_trace:
                    # Sanitize focus question for filename (simple sanitization)
                    focus_q_short = "".join(c if c.isalnum() else "_" for c in training_data[i]['question'][:30])
                    global_output_filename = f"global_reasoning_trace_doc_{i}_{focus_q_short}.txt"
                    global_output_path = os.path.join(global_traces_dir, global_output_filename) # Save in subfolder
                    save_reasoning_trace_global(reasoning_trace, global_output_path)
                else:
                    print(f"No reasoning trace generated or an error occurred for document index {i}.")
    
    print("\nAugmentation process complete.")
    print(f"Results, if any, are in '{args.output_dir}'.")

if __name__ == "__main__":
    asyncio.run(main())
