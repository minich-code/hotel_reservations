

import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from src.hotel_reservations.logger import logger
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.config_manager.config_settings import ConfigurationManager
from src.hotel_reservations.components.c_02_data_validation import DataValidation


PIPELINE_NAME = "DATA VALIDATION PIPELINE"

class DataValidationPipeline:
    def __init__(self): 
        pass

    def run(self):
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        validation_status = data_validation.validate_all_columns()

        if validation_status:
            print("Data validation successful!")
        else:
            print("Data validation failed.")



if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise CustomException(e, sys)
