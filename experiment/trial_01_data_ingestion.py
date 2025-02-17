
import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys

from src.hotel_reservations.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.hotel_reservations.exception import CustomException  
from src.hotel_reservations.logger import logger  
from src.hotel_reservations.utils.commons import create_directories, read_yaml 

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_file: str
    output_file: str

class ConfigurationManager:
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        try:
            self.ingestion_config = read_yaml(data_ingestion_config)
            create_directories([self.ingestion_config['artifacts_root']]) 
        except Exception as e:
            logger.exception(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            return DataIngestionConfig(**data_config)
        except Exception as e:
            logger.exception(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def import_data_from_csv(self) -> Path:  # Return Path to the file
        try:
            df = self._read_data()

            if df.empty:
                logger.warning("No data found in the CSV file.")
                return None

            cleaned_data = self._clean_data(df)
            return self._save_data(cleaned_data)  # Return path to saved file

        except Exception as e:
            logger.exception(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

    def _read_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.source_file)
            return df
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.config.source_file}")
            raise CustomException(f"CSV file not found: {self.config.source_file}", sys)
        except Exception as e:
            logger.exception(f"Error reading data: {e}")
            raise CustomException(e, sys)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

            nunique = df.nunique()
            cols_to_drop = nunique[nunique == 1].index
            df = df.drop(cols_to_drop, axis=1)

            zero_variance_cols = [col for col in numerical_cols if df[col].var() == 0]
            df = df.drop(columns=zero_variance_cols, axis=1)

            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            df.dropna(inplace=True)

            for col in numerical_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass #ignore column

            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.exception(f"Error during data cleaning: {e}")
            raise CustomException(e, sys)

    def _save_data(self, df: pd.DataFrame) -> Path:
        try:
            root_dir = self.config.root_dir
            output_path = Path(root_dir) / self.config.output_file
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.exception(f"Error saving data: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        file_path = data_ingestion.import_data_from_csv()  # Get the file path
        if file_path:
            print(f"Data ingestion process completed successfully. File saved to: {file_path}")
        else:
            print("Data ingestion process completed, but no file was saved (likely due to empty dataset).")

    except Exception as e:
        print(f"Error during data ingestion: {e}")