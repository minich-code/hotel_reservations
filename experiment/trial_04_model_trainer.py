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
from src.hotel_reservations.utils.commons import create_directories, read_yaml
from src.hotel_reservations.constants import MODEL_TRAINER_CONFIG_FILEPATH, PARAMS_CONFIG_FILEPATH

@dataclass
class ModelTrainerConfig:
    root_dir: str
    train_features_path: str
    train_targets_path: str
    model_name: str
    model_params: Dict[str, Any]

class ConfigurationManager:
    def __init__( self,
        model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
        model_params_config: Path = PARAMS_CONFIG_FILEPATH,):
        try:
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)
            create_directories([self.training_config.artifacts_root])
        except Exception as e:
            logger.error(f"Error loading model training config file: {str(e)}")
            raise CustomException(e, sys)

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config['model_trainer']
            model_params = self.model_params_config['XGBClassifier_params']
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir = Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params
            )
            
        except Exception as e:
            logger.exception(f"Error loading model training configuration: {e}")
            raise CustomException(e, sys)
        

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

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(config=model_training_config)
        model_trainer.train()
        logger.info("Model training process completed successfully.")

    except CustomException as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)