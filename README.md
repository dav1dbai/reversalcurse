# Reversal Curse Research

Code, datasets, and experiments for studying the "reversal curse" in language models - a phenomenon where models struggle to invert learned relationships (e.g., "X is Y's parent" vs "Y is X's child").

## Project Structure

```
├── dataset/                  # Dataset generation tools
│   ├── generator.py          # Creates relationship graphs and datasets
│   ├── reverse.py            # Implements relation reversal techniques
│   ├── config*.yaml          # Configuration files
│   ├── data/                 # Base characters and relations
│   ├── completions_*/        # Completion-based datasets
│   └── qa_*/                 # Question-answering datasets
│
├── scripts/                  # Training and evaluation
│   ├── LORA_finetuner.py     # Fine-tune models using LoRA
│   ├── inference.py          # Evaluate model performance
│   ├── comp_inference.py     # Specialized inference for completions
│   └── run_lora_experiments.sh # Run multiple experiments
│
├── logs/                     # Experiment results and metrics
│
└── analyze_*.py              # Error analysis scripts
```

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Generate Dataset
```bash
python dataset/generator.py dataset/config.yaml qa_output_directory
```

### Fine-tune Model
```bash
python scripts/LORA_finetuner.py --lora_rank 512 --base_model_name "Qwen/Qwen2.5-7B-Instruct"
```

### Evaluate
```bash
python scripts/inference.py --model_path "models/qwen7b2.5it_512qasn" --base_model_name "Qwen/Qwen2.5-7B-Instruct"
```

### Analyze Errors
```bash
python analyze_errors.py
```

## Dataset Types
- **QA Format**: "Who is X's mother? Answer: Y"
- **Completion Format**: "X's mother is Y"

## Dataset Configurations
- **SN**: Simple Neutral (simple relationship graphs with gender-neutral relations)
- **SG**: Simple Gendered (simple relationship graphs with gendered relations)
- **CN**: Complex Neutral (complex relationship graphs with gender-neutral relations)
- **CG**: Complex Gendered (complex relationship graphs with gendered relations)
- **_aug**: Datasets with train-time augmentation (reversed variations)