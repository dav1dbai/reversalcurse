#!/bin/bash
# --- Configuration ---
BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
SHORT_MODEL_PREFIX="qwen7b2.5it"
DATASET_SUFFIX="qasn"
LORA_RANKS=(8 16 32 64 128 256 512 1024)

echo "Starting inference runs for all specified LoRA ranks..."
echo "Base Model: $BASE_MODEL_NAME"
echo "Model Prefix: $SHORT_MODEL_PREFIX"
echo "Dataset Suffix: $DATASET_SUFFIX"
echo "LoRA Ranks: ${LORA_RANKS[*]}"

for rank in "${LORA_RANKS[@]}"; do
    MODEL_DIR_BASENAME="${SHORT_MODEL_PREFIX}_${rank}${DATASET_SUFFIX}"
    MODEL_PATH_ARG="models/${MODEL_DIR_BASENAME}"

    echo ""
    echo "----------------------------------------------------"
    echo "Processing LoRA Rank: $rank"
    echo "Adapter Path: $MODEL_PATH_ARG"
    echo "----------------------------------------------------"

    # Check if model directory exists
    if [ ! -d "$MODEL_PATH_ARG" ]; then
        echo "Warning: Model directory '$MODEL_PATH_ARG' does not exist. Skipping."
        continue
    fi

    echo "Command: python inference.py --model_path \"$MODEL_PATH_ARG\" --base_model_name \"$BASE_MODEL_NAME\"" # Add $INFERENCE_EXTRA_ARGS here if using it
    
    # Execute inference script
    python inference.py --model_path "$MODEL_PATH_ARG" --base_model_name "$BASE_MODEL_NAME" # Add $INFERENCE_EXTRA_ARGS here if using it

    echo "[Rank $rank] Inference complete. Results logged by inference.py."
done

echo ""
echo "----------------------------------------------------"
echo "All inference runs completed at $(date)."
echo "Log files should be in the '../logs' directory (relative to scripts directory)."
echo "----------------------------------------------------" 