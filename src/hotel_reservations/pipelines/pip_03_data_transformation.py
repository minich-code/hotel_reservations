



import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from src.hotel_reservations.logger import logger
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.config_manager.config_settings import ConfigurationManager
from src.hotel_reservations.components.c_03_data_transformation import DataTransformation


PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class DataTransformationPipeline:
    def __init__(self): 
        pass

    def run(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_split_data()
        logger.info("Data transformation and splitting completed.")



if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_validation_pipeline = DataTransformationPipeline()
        data_validation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise CustomException(e, sys)
