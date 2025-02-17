

import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib

import sys
from typing import Dict, Any
from xgboost import XGBClassifier
# Local Modules
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.logger import logger
from src.hotel_reservations.config_entity.config_params import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            logger.info("Starting Model Training")

            # Load Training Data
            try:
                with open(self.config.train_features_path, 'rb') as f:
                    X_train = joblib.load(f)
                y_train = pd.read_parquet(self.config.train_targets_path)
      
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                raise CustomException(f"Error loading training data: {e}", sys)

            # Initialize and Train Model
            logger.info("Initializing and Training XGBoost model")
            model = XGBClassifier(**self.config.model_params)
            model.fit(X_train, y_train)

            # Save Trained Model
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            logger.info("Model Training Completed")

        except Exception as e:
            logger.exception(f"Error during model training: {e}")
            raise CustomException(e, sys)

