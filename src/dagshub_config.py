"""Shared DagsHub / MLflow authentication from .env or environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def configure_dagshub():
    """Load credentials and configure DagsHub + MLflow. Returns (repo_owner, repo_name)."""
    import dagshub
    import mlflow

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

    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return repo_owner, repo_name
