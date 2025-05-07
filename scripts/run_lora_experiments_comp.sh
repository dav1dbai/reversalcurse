#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LORA_RANKS=(128 256 512 1024 2048) # Define LoRA ranks to iterate over
DATASET_SUFFIX="qasn" # Used in naming output directories, consistent with LORA_finetuner.py
USE_CHAT_TEMPLATE=true # true or false. If false, --no_chat_template will be passed to python scripts.

# --- Script ---

echo "Running on host: $(hostname)"
echo "Starting script at: $(date)"

# Extract short model name for directory naming, consistent with LORA_finetuner.py
# e.g., "Qwen/Qwen2.5-7B-Instruct" becomes "Qwen2.5-7B-Instruct"
SHORT_MODEL_NAME="qwen7b2.5it" # Fixed prefix for new naming scheme

echo "Starting LoRA rank experiments..."
echo "Base Model: $BASE_MODEL_NAME"
echo "LoRA Ranks: ${LORA_RANKS[*]}"
echo "Dataset Suffix for naming: $DATASET_SUFFIX"

for rank in "${LORA_RANKS[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "Processing LoRA Rank: $rank"
    echo "----------------------------------------------------"

    # Model output directory basename (e.g., Qwen2.5-7B-Instruct_8_qasn)
    MODEL_SPECIFIC_DIR_BASENAME="${SHORT_MODEL_NAME}_${rank}${DATASET_SUFFIX}"
    # Path relative to the 'scripts' directory (e.g., models/Qwen2.5-7B-Instruct_8_qasn)
    # This is where LORA_finetuner.py will save the adapters.
    TUNER_OUTPUT_MODEL_PATH="models/${MODEL_SPECIFIC_DIR_BASENAME}"

    echo "[Rank $rank] Step 1: Fine-tuning..."
    echo "Command: python LORA_comp_finetuner.py --lora_rank \"$rank\" --base_model_name \"$BASE_MODEL_NAME\""
    
    python LORA_comp_finetuner.py --lora_rank "$rank" --base_model_name "$BASE_MODEL_NAME"

    # LORA_finetuner.py constructs its output_dir like: models/{short_model_name}_{lora_rank}_qasn
    # This TUNER_OUTPUT_MODEL_PATH must match that.

    echo "[Rank $rank] Fine-tuning complete. Adapter saved to $PWD/$TUNER_OUTPUT_MODEL_PATH"

    echo "[Rank $rank] Step 2: Running inference..."
    echo "Command: python comp_inference.py --model_path \\"$TUNER_OUTPUT_MODEL_PATH\\" --base_model_name \\"$BASE_MODEL_NAME\\""
    echo "IMPORTANT: Ensure inference.py is configured to load LoRA adapters (e.g., use_full_model = False)."
    
    echo "Waiting for 10 seconds to ensure model saving is complete..."
    sleep 10
    
    python comp_inference.py --model_path "$TUNER_OUTPUT_MODEL_PATH" --base_model_name "$BASE_MODEL_NAME"

    echo "[Rank $rank] Inference complete. Results logged by inference.py to the main 'logs' directory."
done

echo ""
echo "----------------------------------------------------"
echo "All LoRA rank experiments completed at $(date)."
echo "Log files should be in the 'logs' directory at the workspace root."
echo "CRITICAL REMINDER: Verify that 'scripts/inference.py' was correctly set up to load LoRA adapters (e.g., by setting use_full_model = False) for these experiments."
echo "----------------------------------------------------" 