


from pathlib import Path
import pandas as pd
import numpy as np
import sys


from src.hotel_reservations.exception import CustomException  
from src.hotel_reservations.logger import logger  
from src.hotel_reservations.config_entity.config_params import DataIngestionConfig



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


