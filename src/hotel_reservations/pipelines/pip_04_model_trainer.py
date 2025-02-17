

import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from src.hotel_reservations.logger import logger
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.config_manager.config_settings import ConfigurationManager
from src.hotel_reservations.components.c_04_model_trainer import ModelTrainer


PIPELINE_NAME = "MODEL TRAINING PIPELINE"

class ModelTrainerPipeline:
    def __init__(self): 
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(config=model_training_config)
        model_trainer.train()
        logger.info("Model training process completed successfully.")



if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_validation_pipeline = ModelTrainerPipeline()
        data_validation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise CustomException(e, sys)
