#!/bin/bash
set -e

BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LORA_RANKS=(128 256 512 1024 2048)
DATASET_SUFFIX="qasn"
USE_CHAT_TEMPLATE=true

echo "Running on host: $(hostname)"
echo "Starting script at: $(date)"

SHORT_MODEL_NAME="qwen7b2.5it"

echo "Starting LoRA rank experiments..."
echo "Base Model: $BASE_MODEL_NAME"
echo "LoRA Ranks: ${LORA_RANKS[*]}"
echo "Dataset Suffix for naming: $DATASET_SUFFIX"

for rank in "${LORA_RANKS[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "Processing LoRA Rank: $rank"
    echo "----------------------------------------------------"

    MODEL_SPECIFIC_DIR_BASENAME="${SHORT_MODEL_NAME}_${rank}${DATASET_SUFFIX}"
    TUNER_OUTPUT_MODEL_PATH="models/${MODEL_SPECIFIC_DIR_BASENAME}"

    echo "[Rank $rank] Step 1: Fine-tuning..."
    echo "Command: python LORA_comp_finetuner.py --lora_rank \"$rank\" --base_model_name \"$BASE_MODEL_NAME\""
    
    python LORA_comp_finetuner.py --lora_rank "$rank" --base_model_name "$BASE_MODEL_NAME"

    echo "[Rank $rank] Fine-tuning complete. Adapter saved to $PWD/$TUNER_OUTPUT_MODEL_PATH"

    echo "[Rank $rank] Step 2: Running inference..."
    echo "Command: python comp_inference.py --model_path \"$TUNER_OUTPUT_MODEL_PATH\" --base_model_name \"$BASE_MODEL_NAME\""
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