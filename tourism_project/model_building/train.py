# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score, make_scorer

# for model serialization
import joblib

# for hugging face authentication
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("travel-buyer-prediction-experiment")

api = HfApi()

# Load data
Xtrain = pd.read_csv("hf://datasets/ShabN/visit-with-us-travel-buyer-prediction/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/ShabN/visit-with-us-travel-buyer-prediction/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/ShabN/visit-with-us-travel-buyer-prediction/ytrain.csv").squeeze()
ytest = pd.read_csv("hf://datasets/ShabN/visit-with-us-travel-buyer-prediction/ytest.csv").squeeze()

# Feature lists
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# Handle class imbalance
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (
        OneHotEncoder(drop='first', handle_unknown='ignore'),
        categorical_features
    ),
    remainder='passthrough'
)

# Define XGBoost Classifier (Model)
xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

# Model Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6]
}

# Recall scorer
recall_scorer = make_scorer(recall_score, pos_label=1)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring=recall_scorer, n_jobs=-1) # Grid Search with Cross Validation
    grid_search.fit(Xtrain, ytrain)

# Log CV recall for all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
      with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("cv_mean_recall", results['mean_test_score'][i])
            mlflow.log_metric("cv_std_recall", results['std_test_score'][i])

# Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

# Evaluation
    classification_threshold = 0.45

    y_train_pred = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_test_pred = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_train_pred, output_dict=True)
    test_report = classification_report(ytest, y_test_pred, output_dict=True)

    mlflow.log_metrics({
        "train_recall": train_report['1']['recall'],
        "train_precision": train_report['1']['precision'],
        "train_f1": train_report['1']['f1-score'],
        "test_recall": test_report['1']['recall'],
        "test_precision": test_report['1']['precision'],
        "test_f1": test_report['1']['f1-score']
    })

# Save model
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

# Upload to Hugging Face Space
    repo_id = "ShabN/visit-with-us-travel-buyer-prediction"

try:
  api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
  create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type="model"
    )
