from flask import Flask, render_template, request
import mlflow
import pickle
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import dagshub

try:
    from flask_app.preprocessing_utility import preprocess_text
except ModuleNotFoundError:
    from preprocessing_utility import preprocess_text
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        "DAGSHUB_USER_TOKEN is not set. Add it to mlops_capstone_project/.env"
    )
if not repo_owner or not repo_name:
    raise EnvironmentError(
        "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set in mlops_capstone_project/.env"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(mlflow_tracking_uri)


def load_label_map() -> dict:
    """Load class-id to label mapping from params.yaml or models/label_map.json."""
    label_map_path = PROJECT_ROOT / "models" / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as file:
            return {str(key): value for key, value in json.load(file).items()}

    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, encoding="utf-8") as file:
        params = yaml.safe_load(file)

    label_map = params.get("prediction", {}).get("label_map", {})
    return {str(key): str(value) for key, value in label_map.items()}


def resolve_sentiment_label(prediction) -> tuple[str, str]:
    """Convert a model prediction into a display label and CSS class."""
    prediction_key = str(prediction)
    sentiment_label = LABEL_MAP.get(prediction_key, prediction_key)
    result_class = sentiment_label.lower().replace(" ", "-")
    return sentiment_label, result_class


LABEL_MAP = load_label_map()

app = Flask(__name__)
# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Model and vectorizer setup
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "my_model")


def load_prediction_model():
    """Load model from local artifacts or MLflow registry."""
    local_model_path = PROJECT_ROOT / "models/model.pkl"
    model_source = os.getenv("MODEL_SOURCE", "local").lower()

    if model_source == "local" and local_model_path.exists():
        print(f"Loading model from local file: {local_model_path}")
        with open(local_model_path, "rb") as file:
            return pickle.load(file)

    client = mlflow.MlflowClient()

    for stage in ("Production", "Staging"):
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if versions:
            uri = f"models:/{MODEL_NAME}/{versions[0].version}"
            print(f"Loading model from registry ({stage}): {uri}")
            return mlflow.pyfunc.load_model(uri)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest = max(versions, key=lambda version: int(version.version))
        uri = f"models:/{MODEL_NAME}/{latest.version}"
        print(f"Loading model from registry (latest): {uri}")
        return mlflow.pyfunc.load_model(uri)

    experiment_info_path = PROJECT_ROOT / "reports/experiment_info.json"
    if experiment_info_path.exists():
        with open(experiment_info_path, encoding="utf-8") as file:
            model_info = json.load(file)
        uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        print(f"Loading model from MLflow run: {uri}")
        return mlflow.pyfunc.load_model(uri)

    if local_model_path.exists():
        print(f"Loading model from local file: {local_model_path}")
        with open(local_model_path, "rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(
        f"No model found for '{MODEL_NAME}'. Run the DVC pipeline or register the model in MLflow."
    )


model = load_prediction_model()
vectorizer_path = PROJECT_ROOT / "models/vectorizer.pkl"
if not vectorizer_path.exists():
    raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)
# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", sentiment_label=None, result_class=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    text = preprocess_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray())

    prediction = model.predict(features_df)[0]
    sentiment_label, result_class = resolve_sentiment_label(prediction)

    PREDICTION_COUNT.labels(prediction=sentiment_label).inc()

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template(
        "index.html",
        sentiment_label=sentiment_label,
        result_class=result_class,
    )
@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker