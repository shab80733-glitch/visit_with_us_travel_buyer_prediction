from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("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"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="ShabN/visit-with-us-travel-buyer-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
