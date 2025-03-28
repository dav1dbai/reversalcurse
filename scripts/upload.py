from huggingface_hub import HfApi, create_repo
import os

# Define paths and repository info
adapter_path = "models/reversal_curse_7b_rank512"  # Path to your saved LoRA adapter
repo_id = "dav1dbai/qwen-reversal-curse-lora"  # Your HF username and desired repo name

# Create the repository
create_repo(repo_id, private=True, exist_ok=True)

# Initialize the Hugging Face API
api = HfApi()

# Upload the adapter files directly from the adapter directory
api.upload_folder(
    folder_path=adapter_path,
    repo_id=repo_id,
    repo_type="model"
)

print(f"Successfully uploaded adapter to https://huggingface.co/{repo_id}")