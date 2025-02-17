

import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from src.hotel_reservations.logger import logger
from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.config_manager.config_settings import ConfigurationManager
from src.hotel_reservations.components.c_01_data_ingestion import DataIngestion

PIPELINE_NAME = "DATA INGESTION PIPELINE"

class DataIngestionPipeline:
    def __init__(self): 
        pass

    def run(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        file_path = data_ingestion.import_data_from_csv()  # Get the file path
        if file_path:
            print(f"Data ingestion process completed successfully. File saved to: {file_path}")
        else:
            print("Data ingestion process completed, but no file was saved (likely due to empty dataset).")


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise CustomException(e, sys)