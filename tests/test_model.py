# load test + signature test + performance test

import os
import pickle
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dagshub_config import PROJECT_ROOT, configure_dagshub

configure_dagshub()
import mlflow


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        if not cls.new_model_version:
            raise RuntimeError(
                f"No model found in Staging for '{cls.new_model_name}'. "
                "Run register_model.py or dvc repro first."
            )

        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        with open(PROJECT_ROOT / "models" / "vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_bow.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])],
        )

        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f"Accuracy should be at least {expected_accuracy}")
        self.assertGreaterEqual(precision_new, expected_precision, f"Precision should be at least {expected_precision}")
        self.assertGreaterEqual(recall_new, expected_recall, f"Recall should be at least {expected_recall}")
        self.assertGreaterEqual(f1_new, expected_f1, f"F1 score should be at least {expected_f1}")


if __name__ == "__main__":
    unittest.main()
