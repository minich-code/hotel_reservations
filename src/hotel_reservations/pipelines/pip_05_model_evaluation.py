

import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from src.hotel_reservations.logger import logger
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.config_manager.config_settings import ConfigurationManager
from src.hotel_reservations.components.c_05_model_evaluation import ModelEvaluation 


PIPELINE_NAME = "MODEL EVALUATION PIPELINE"

class ModelEvaluationPipeline:
    def __init__(self): 
        pass

    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run_validation()



if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise CustomException(e, sys)
