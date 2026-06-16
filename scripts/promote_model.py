# promote model

from src.dagshub_config import configure_dagshub

import mlflow


def promote_model():
    configure_dagshub()

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
