import sys

sys.path.append('/home/western/DS_Projects/hotel_reservations')

from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.logger import logger
from src.hotel_reservations.utils.commons import create_directories, read_yaml
from src.hotel_reservations.config_entity.config_params import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        logger.info("Loading model")
        try:
            with open(self.config.model_path, 'rb') as f:
                model = joblib.load(f)
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def load_data(self):
        """Loads the test data from the given file paths."""
        logger.info("Loading test data")
        try:
            with open(self.config.test_feature_path, 'rb') as f:
                X_test = joblib.load(f)
            y_test = pd.read_parquet(self.config.test_targets_path)

            logger.info("Test data loaded successfully")
            return X_test, y_test

        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluates the model and saves evaluation metrics."""
        try:
            logger.info("Making predictions and generating evaluation metrics")
            y_pred = model.predict(X_test)

            # Generate and save classification report
            class_report = classification_report(y_test, y_pred, zero_division=0)
            with open(self.config.classification_report_path, "w") as f:
                f.write(class_report)
            logger.info(f"Classification report saved to {self.config.classification_report_path}")

            # Save Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(self.config.confusion_matrix_path)
            plt.close()
            logger.info(f"Confusion matrix saved to {self.config.confusion_matrix_path}")

            # Generate ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(self.config.roc_curve_path)
            plt.close()
            logger.info(f"ROC curve saved to {self.config.roc_curve_path}")

            # Generate Precision-Recall Curve
            precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
            pr_auc = auc(recall, precision)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.savefig(self.config.pr_curve_path)
            plt.close()
            logger.info(f"Precision-recall curve saved to {self.config.pr_curve_path}")

            logger.info("All evaluation metrics generated and saved successfully")

            eval_scores = {
                "roc_auc":roc_auc,
                "pr_auc":pr_auc
            }

            with open(self.config.eval_scores_path, 'w') as f:
                json.dump(eval_scores, f)
            logger.info(f"Data evaluation process completed successfully and saved to: {self.config.eval_scores_path}")


        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(e, sys)

    def run_validation(self):
        """Runs the model validation process."""
        try:
            X_test, y_test = self.load_data()
            model = self.load_model()
            self.evaluate_model(model, X_test, y_test)
            logger.info("Model validation completed successfully.")
        except Exception as e:
            logger.error(f"Error during the model validation process: {e}")
            raise CustomException(f"Error during the model validation process: {e}", sys)

