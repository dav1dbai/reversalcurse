import csv
import argparse
import random
import sys
import os

# Attempt to import spacy, but don't fail immediately if not needed
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    # print("Warning: spacy not installed. 'entity' method will not be available.", file=sys.stderr)

# Global variable for spaCy model to avoid reloading
NLP_MODEL = None

def load_spacy_model(model_name="en_core_web_sm"):
    """Loads the spaCy model."""
    global NLP_MODEL
    if not SPACY_AVAILABLE:
        print("Error: spaCy library not found. Please install it (`pip install spacy`)", file=sys.stderr)
        print("       and download a model (`python -m spacy download en_core_web_sm`)", file=sys.stderr)
        sys.exit(1)
    if NLP_MODEL is None:
        try:
            print(f"Loading spaCy model '{model_name}'...")
            NLP_MODEL = spacy.load(model_name)
            print("Model loaded.")
        except OSError:
            print(f"Error: spaCy model '{model_name}' not found.", file=sys.stderr)
            print(f"Please download it using: python -m spacy download {model_name}", file=sys.stderr)
            sys.exit(1)
    return NLP_MODEL

def reverse_word(text: str) -> str:
    """
    Reverses the order of words in a string.
    Example: "Who is the patient?" -> "patient? the is Who"
    """
    words = text.split()
    return " ".join(words[::-1])

def reverse_entity(text: str) -> str:
    """
    Reverses the order of tokens, treating named entities as single units.
    Requires spaCy and a language model.
    Example: "Who is the patient of Liam O'Connor?" -> "Liam O'Connor of patient the is Who"
             "Malik Johnson is the patient of Liam O'Connor" -> "Liam O'Connor of patient the is Malik Johnson"
    """
    nlp = load_spacy_model() # Load if not already loaded
    doc = nlp(text)
    elements = []
    
    # Keep track of the end position of the last processed entity
    last_entity_end = 0 
    
    for ent in doc.ents:
        # Add non-entity tokens before this entity
        non_entity_part = doc[last_entity_end:ent.start].text.strip()
        if non_entity_part:
            elements.extend(non_entity_part.split()) # Split into words
            
        # Add the whole entity as one element
        elements.append(ent.text)
        
        # Update the position after this entity
        last_entity_end = ent.end
        
    # Add any remaining non-entity tokens after the last entity
    remaining_part = doc[last_entity_end:].text.strip()
    if remaining_part:
        elements.extend(remaining_part.split()) # Split into words

    # Reverse the list of elements (entity texts and non-entity words)
    reversed_elements = elements[::-1]
    
    # Join elements
    return " ".join(reversed_elements)


def reverse_random_segment(text: str, k: int) -> str:
    """
    Splits text into random segments (size 1 to k), reverses the order
    of segments, and joins with '[REV]'. Word order within segments is preserved.
    Example (k=3): "Who is the patient of Liam?" -> (Segments: ["Who is"], ["the", "patient of"], ["Liam?"])
                                               -> "Liam?[REV]the patient of[REV]Who is" (randomness applies)
    """
    words = text.split()
    if not words:
        return ""

    segments = []
    current_pos = 0
    while current_pos < len(words):
        # Determine segment size: 1 to k, but not exceeding remaining words
        max_size = min(k, len(words) - current_pos)
        size = random.randint(1, max_size)
        segment = words[current_pos : current_pos + size]
        segments.append(" ".join(segment))
        current_pos += size

    # Reverse the order of the segments
    reversed_segments = segments[::-1]
    return "[REV]".join(reversed_segments)

def process_csv(input_file: str, output_file: str, method: str, k: int = 5):
    """
    Reads the input CSV (assuming one sequence per row in the first column),
    applies the specified reversal method to the sequence, and writes the
    result to the output CSV (one reversed sequence per row).
    """
    print(f"Processing '{input_file}' using method '{method}'...")
    if method == 'random':
        print(f"Using max segment size k={k} for random reversal.")

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Optional: Write a header for the output file
            # writer.writerow(['reversed_sequence'])

            processed_count = 0
            for row in reader:
                if not row: # Skip empty rows
                    continue
                original_sequence = row[0] # Assume sequence is in the first column
                try:
                    if method == 'word':
                        modified_sequence = reverse_word(original_sequence)
                    elif method == 'entity':
                        # Ensure spaCy is loaded if needed for this method
                        load_spacy_model()
                        modified_sequence = reverse_entity(original_sequence)
                    elif method == 'random':
                        modified_sequence = reverse_random_segment(original_sequence, k)
                    else:
                        # Should not happen due to argparse choices
                        print(f"Error: Unknown method '{method}'", file=sys.stderr)
                        return # Or sys.exit(1)

                    writer.writerow([modified_sequence]) # Write the reversed sequence as a single column row
                    processed_count += 1
                except Exception as e:
                    print(f"\nError processing row: {row}", file=sys.stderr)
                    print(f"Error message: {e}", file=sys.stderr)
                    # Optionally skip the row or exit
                    # continue
                    sys.exit(1) # Exit on first error for safety

            print(f"\nProcessing complete. {processed_count} rows processed.")
            print(f"Output saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)
    except IndexError:
         print(f"\nError: Input CSV file '{input_file}' seems to have empty rows or rows without data in the first column.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply reversal techniques to sequences in a CSV file (one sequence per row, first column).")
    parser.add_argument("input_csv", help="Path to the input CSV file (expects one sequence per row in the first column).")
    parser.add_argument("output_csv", help="Path to save the modified CSV file.")
    parser.add_argument(
        "--method",
        required=True,
        choices=['word', 'entity', 'random'],
        help="Reversal method to apply: 'word' (REVERSEword), 'entity' (REVERSEentity), 'random' (REVERSErand)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Maximum segment size for 'random' method (REVERSErand). Default: 5"
    )
    # Optional: Add argument for spaCy model name if needed
    # parser.add_argument("--spacy_model", default="en_core_web_sm", help="Name of the spaCy model to use for 'entity' method.")

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.input_csv):
         print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
         sys.exit(1)
    if args.method == 'entity' and not SPACY_AVAILABLE:
         print("Error: 'entity' method requires spaCy, but it's not installed.", file=sys.stderr)
         print("Please install it (`pip install spacy`) and download a model.", file=sys.stderr)
         sys.exit(1)
    if args.k <= 0 and args.method == 'random':
        print("Error: --k must be a positive integer for the 'random' method.", file=sys.stderr)
        sys.exit(1)


    process_csv(args.input_csv, args.output_csv, args.method, args.k)
