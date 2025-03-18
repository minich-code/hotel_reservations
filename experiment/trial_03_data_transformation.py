
import sys
sys.path.append('/home/western/ds_projects/hotel_reservations')


from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.logger import logger
from src.hotel_reservations.utils.commons import create_directories, read_yaml, save_object
from src.hotel_reservations.constants import DATA_TRANSFORMATION_CONFIG_FILEPATH


@dataclass
class DataTransformationConfig:
    root_dir: str
    data_path: str
    numerical_cols: list
    categorical_cols: list
    target_col: str
    random_state: int
    test_size: float

class ConfigurationManager:
    def __init__(self, data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH):
        try:
            self.preprocessing_config = read_yaml(data_preprocessing_config)
            create_directories([self.preprocessing_config['artifacts_root']])
        except Exception as e:
            logger.exception(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            
            transformation_config = self.preprocessing_config['data_transformation']
            create_directories([transformation_config['root_dir']])

            data_transformation_config = DataTransformationConfig(
                root_dir=transformation_config['root_dir'],
                data_path=transformation_config['data_path'],
                numerical_cols=transformation_config['numerical_cols'],
                categorical_cols=transformation_config['categorical_cols'],
                target_col=transformation_config['target_col'],
                random_state=transformation_config['random_state'],
                test_size = transformation_config['test_size']
                )
            
            return data_transformation_config
        except Exception as e:
            logger.exception(f"Error getting Data Transformation config: {e}")
            raise CustomException(e, sys)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_object(self) -> ColumnTransformer:
        logger.info("Creating transformer object")

        try:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.config.numerical_cols),
                    ('cat', categorical_transformer, self.config.categorical_cols),
                ], remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            logger.exception(f"Error creating transformer object: {str(e)}")
            raise CustomException(e, sys)

    def train_test_split_data(self) -> None:
        try:
            logger.info("Splitting data into train and test sets")

            # Load data
            data_path = self.config.data_path
            try:
                with open(data_path, 'rb') as f:
                    df = pd.read_parquet(f)
                logger.info(f"Data shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading Parquet file: {e}")
                raise CustomException(f"Error reading Parquet file: {e}", sys)

            # Define features (X) and target (y)
            X = df.drop(self.config.target_col, axis=1)
            y = df[self.config.target_col]

            # Encode target variable using LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info(f"Target variable '{self.config.target_col}' label encoded.")

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, stratify=y, random_state=self.config.random_state
            )

            logger.info("Data splitting completed.")

            # Get preprocessor object
            preprocessor = self.get_transformer_object()

            # Fit and transform training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            logger.info("Training data transformation completed.")

            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            logger.info("Test data transformation completed.")
           
            transformed_data_dir = Path(self.config.root_dir) #Gets directory root path
            
            # Saving objects
            save_object(obj=preprocessor, file_path=transformed_data_dir / 'preprocessor.joblib') #Used saveObject instead of joblib

            pd.DataFrame(y_train).to_parquet(transformed_data_dir / 'y_train.parquet', index=False) #Changed to Frame
            pd.DataFrame(y_test).to_parquet(transformed_data_dir / 'y_test.parquet', index=False)

            joblib.dump(X_train_transformed, transformed_data_dir / 'X_train_transformed.joblib')
            joblib.dump(X_test_transformed, transformed_data_dir / 'X_test_transformed.joblib')

            logger.info("All transformed data and preprocessor saved successfully.")

        except Exception as e:
            logger.exception(f"Error during data transformation: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_split_data()
        logger.info("Data transformation and splitting completed.")

    except CustomException as e:
        logger.error(f"Error in data transformation: {str(e)}")
        sys.exit(1)