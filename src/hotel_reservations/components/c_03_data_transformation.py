


import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

import sys

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib
import os
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.hotel_reservations.exception import CustomException
from src.hotel_reservations.logger import logger
from src.hotel_reservations.utils.commons import save_object
from src.hotel_reservations.config_entity.config_params import DataTransformationConfig


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

            # Split into features (X) and target (y)
            X = df.drop(self.config.target_col, axis=1)
            y = df[self.config.target_col]

            # Encode target variable using LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info(f"Target variable '{self.config.target_col}' label encoded.")

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.config.random_state
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

