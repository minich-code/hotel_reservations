import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import os
import sys
import json
import yaml

from src.hotel_reservations.exception import CustomException  # Import custom exception
from src.hotel_reservations.logger import logger  # Import custom logger
from src.hotel_reservations.utils.commons import create_directories, read_yaml
from src.hotel_reservations.constants import DATA_VALIDATION_CONFIG_FILEPATH

@dataclass
class DataValidationConfig:
    root_dir: str
    data_dir: str
    val_status: str
    all_schema: dict
    validated_data: str  

class ConfigurationManager:
    def __init__(self, config_filepath: str = DATA_VALIDATION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([self.config['artifact_root']])  # Corrected key
        except Exception as e:
            logger.exception(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config['data_validation']
            create_directories([config['root_dir']])

            data_validation_config = DataValidationConfig(
                root_dir=config['root_dir'],
                data_dir=config['data_dir'],
                val_status=config['val_status'],
                all_schema=config['all_schema'],
                validated_data=config['validated_data']
            )
            return data_validation_config
        except Exception as e:
            logger.exception(f"Error getting Data Validation config: {e}")
            raise CustomException(e, sys)

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            overall_status = True  # Assume valid until proven otherwise
            validation_results = {} #Collects all validation details

            try:
                data = pd.read_parquet(self.config.data_dir)  # Read the Parquet file
            except Exception as e:
                logger.error(f"Error reading Parquet file: {e}")
                raise CustomException(f"Error reading Parquet file: {e}", sys)

            all_cols = list(data.columns)
            all_schema = self.config.all_schema

            for col in all_cols:
                if col not in all_schema:
                    logger.error(f"Column {col} not found in schema")
                    validation_results[col] = "Column missing in schema"
                    overall_status = False #No need to continue, validation failed
                else:
                    validation_results[col] = "Column present in schema"

            #Additional check for column datatypes
            if overall_status:
                for col in all_cols:
                    expected_dtype = str(all_schema[col])
                    actual_dtype = str(data[col].dtype)
                    if expected_dtype != actual_dtype:
                        logger.error(f"Column {col} has incorrect data type: expected {expected_dtype}, got {actual_dtype}")
                        validation_results[col] = f"Incorrect data type: expected {expected_dtype}, got {actual_dtype}"
                        overall_status = False
                    else:
                        validation_results[col] = "Data type valid"

            # Save results to a file
            val_status_path = self.config.val_status
            try:
                with open(val_status_path, 'w') as f:
                    json.dump(validation_results, f, indent=4)
                logger.info(f"Validation results saved to {val_status_path}")
            except Exception as e:
                logger.error(f"Failed to save validation results: {e}")
                raise CustomException(f"Failed to save validation results: {e}", sys)


            root_dir_path = Path(self.config.root_dir)

            # Save the data to a parquet file only if the validation passed
            if overall_status:
                try:
                    output_path = self.config.validated_data
                    data.to_parquet(output_path, index=False) #use to_parquet
                    logger.info(f"Validated data saved to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save validated data: {e}")
                    raise CustomException(f"Failed to save validated data: {e}", sys)
            else:
                logger.warning(f"Data validation failed. Check {val_status_path} for more details")
                

            return overall_status

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        validation_status = data_validation.validate_all_columns()

        if validation_status:
            print("Data validation successful!")
        else:
            print("Data validation failed.")

    except Exception as e:
        print(f"Error: {e}")