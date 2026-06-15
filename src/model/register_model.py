# register model

import json
import os
from pathlib import Path

import dagshub
import mlflow
from dotenv import load_dotenv
import src.logger  # configures logging handlers
import logging

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")
mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow",
)

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_USER_TOKEN is not set. Add it to .env locally or GitHub Actions secrets in CI."
    )
if not repo_owner or not repo_name:
    raise EnvironmentError(
        "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set in .env or GitHub Actions secrets."
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(mlflow_tracking_uri)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()