from huggingface_hub import HfApi, create_repo
import os

adapter_path = "models/reversal_curse_7b_rank512"
repo_id = "davidbai/qwen-reversal-curse-lora"

create_repo(repo_id, private=True, exist_ok=True)

api = HfApi()

api.upload_folder(
    folder_path=adapter_path,
    repo_id=repo_id,
    repo_type="model"
)

print(f"Successfully uploaded adapter to https://huggingface.co/{repo_id}")