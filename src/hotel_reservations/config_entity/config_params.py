
from dataclasses import dataclass
from typing import Dict, Any


# Data Ingestion
@dataclass
class DataIngestionConfig:
    root_dir: str
    source_file: str
    output_file: str

# Data Validation
@dataclass
class DataValidationConfig:
    root_dir: str
    data_dir: str
    val_status: str
    all_schema: dict
    validated_data: str  


# Data Transformation 
@dataclass
class DataTransformationConfig:
    root_dir: str
    data_path: str
    numerical_cols: list
    categorical_cols: list
    target_col: str
    random_state: int

# Model Training 

@dataclass
class ModelTrainerConfig:
    root_dir: str
    train_features_path: str
    train_targets_path: str
    model_name: str
    model_params: Dict[str, Any]

# Model Evaluation
@dataclass
class ModelEvaluationConfig:
    root_dir: str
    test_feature_path: str
    test_targets_path: str
    model_path: str
    eval_scores_path: str
    classification_report_path: str
    confusion_matrix_path: str
    roc_curve_path: str
    pr_curve_path: str
