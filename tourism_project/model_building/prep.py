# for data manipulation
import pandas as pd
import sklearn
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/ShabN/visit-with-us-travel-buyer-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop non-informative columns
df.drop(columns=[ 'CustomerID'], inplace=True)


# Manual Ordinal Encoding
designation_map = {
    'Executive': 1,
    'Manager': 2,
    'Senior Manager': 3,
    'AVP': 4,
    'VP': 5
}

df['Designation'] = df['Designation'].map(designation_map)

# Target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="ShabN/visit-with-us-travel-buyer-prediction",
        repo_type="dataset",
    )
