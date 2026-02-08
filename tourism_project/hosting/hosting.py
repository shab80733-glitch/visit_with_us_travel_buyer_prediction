from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_ADML_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="ShabN/visit-with-us-buyer-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
