#!/bin/bash
# --- Configuration ---
BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR_PREFIX="qwen7b"
MODEL_DIR_SUFFIX_AFTER_RANK="_comp_sn"
LORA_RANKS=(128 256 512 1024)
DATASET_NAME_ARG="completions_sn"

echo "Starting inference runs for all specified LoRA ranks..."
echo "Base Model: $BASE_MODEL_NAME"
echo "Model Directory Prefix: $MODEL_DIR_PREFIX"
echo "Model Directory Suffix after Rank: $MODEL_DIR_SUFFIX_AFTER_RANK"
echo "LoRA Ranks: ${LORA_RANKS[*]}"
echo "Dataset Name for Inference Script: $DATASET_NAME_ARG"

for rank in "${LORA_RANKS[@]}"; do
    MODEL_DIR_BASENAME="${MODEL_DIR_PREFIX}_${rank}${MODEL_DIR_SUFFIX_AFTER_RANK}"
    MODEL_PATH_ARG="models/${MODEL_DIR_BASENAME}"

    echo ""
    echo "----------------------------------------------------"
    echo "Processing LoRA Rank: $rank"
    echo "Adapter Path: $MODEL_PATH_ARG"
    echo "----------------------------------------------------"

    if [ ! -d "$MODEL_PATH_ARG" ]; then
        echo "Warning: Model directory '$MODEL_PATH_ARG' does not exist. Skipping."
        continue
    fi

    echo "Command: python comp_inference.py --model_path \"$MODEL_PATH_ARG\" --base_model_name \"$BASE_MODEL_NAME\" --dataset_name \"$DATASET_NAME_ARG\""
    
    python comp_inference.py --model_path "$MODEL_PATH_ARG" --base_model_name "$BASE_MODEL_NAME" --dataset_name "$DATASET_NAME_ARG"

    echo "[Rank $rank] Inference complete. Results logged by comp_inference.py."
done

echo ""
echo "----------------------------------------------------"
echo "All inference runs completed at $(date)."
echo "Log files should be in the '../logs' directory (relative to scripts directory)."
echo "----------------------------------------------------" 