from huggingface_hub import HfApi
import os

print(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="models",
    repo_id="AdityaPugalia/text-emotion-recognition",
    repo_type="model",
    path_in_repo='models/'
)
