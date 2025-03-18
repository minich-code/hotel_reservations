



import sys

from src.hotel_reservations.constants import *
from src.hotel_reservations.exception import CustomException  
from src.hotel_reservations.logger import logger  
from src.hotel_reservations.utils.commons import create_directories, read_yaml 
from src.hotel_reservations.config_entity.config_params import *


class ConfigurationManager:
    def __init__(
            self, 
            data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH,
            config_filepath: str = DATA_VALIDATION_CONFIG_FILEPATH,
            data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH,
            model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
            model_params_config: Path = PARAMS_CONFIG_FILEPATH,
            model_evaluation_config: str = MODEL_EVALUATION_CONFIG_FILEPATH
            ):

        try:
            
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.config = read_yaml(config_filepath)
            self.preprocessing_config = read_yaml(data_preprocessing_config)
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)
            self.evaluation_config = read_yaml(model_evaluation_config)

            create_directories([self.ingestion_config['artifacts_root']]) 
            create_directories([self.config['artifact_root']])
            create_directories([self.preprocessing_config['artifacts_root']])
            create_directories([self.training_config.artifacts_root])
            create_directories([self.evaluation_config.artifacts_root])
        


        except Exception as e:
            logger.exception(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

# Data Ingestion 
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:

            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            return DataIngestionConfig(**data_config)
        
        except Exception as e:
            logger.exception(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
        
# Data Validation 
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
        
        
# Data Transformation      
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
                test_size=transformation_config['test_size']
            )
            return data_transformation_config
        except Exception as e:
            logger.exception(f"Error getting Data Transformation config: {e}")
            raise CustomException(e, sys)


# Model Training 
    def get_model_training_config(self) -> ModelTrainerConfig:
        try:
            trainer_config = self.training_config['model_trainer']
            create_directories([trainer_config.root_dir])

            model_params = self.model_params_config['XGBClassifier_params']

            return ModelTrainerConfig(
                root_dir = trainer_config.root_dir,
                train_features_path=trainer_config.train_features_path,
                train_targets_path=trainer_config.train_targets_path,
                model_name=trainer_config.model_name,
                model_params=model_params
            )
            
        except Exception as e:
            logger.exception(f"Error loading model training configuration: {e}")
            raise CustomException(e, sys)
# Model Evaluation 
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Getting model evaluation configuration")
        eval_config = self.evaluation_config['model_evaluation']
        create_directories([eval_config['root_dir']])

        return ModelEvaluationConfig(
            root_dir=eval_config['root_dir'],
            test_feature_path=eval_config['test_feature_path'],
            test_targets_path=eval_config['test_targets_path'],
            model_path=eval_config['model_path'],
            eval_scores_path=eval_config['eval_scores_path'],
            classification_report_path=eval_config['classification_report_path'],
            confusion_matrix_path=eval_config['confusion_matrix_path'],
            roc_curve_path=eval_config['roc_curve_path'],
            pr_curve_path=eval_config['pr_curve_path']
        )