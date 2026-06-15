# load test + signature test + performance test

import unittest
import mlflow
import os
import pickle
from pathlib import Path

import dagshub
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()