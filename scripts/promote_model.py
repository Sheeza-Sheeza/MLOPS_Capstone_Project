# promote model

import os
from pathlib import Path

import dagshub
import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def promote_model():
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

    client = mlflow.MlflowClient()

    model_name = "my_model"
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived",
        )

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production",
    )
    print(f"Model version {latest_version_staging} promoted to Production")


if __name__ == "__main__":
    promote_model()
