
import sys
sys.path.append('/home/western/DS_Projects/hotel_reservations')

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys
import joblib


from src.hotel_reservations.logger import logger
from src.hotel_reservations.constants import DATA_TRANSFORMATION_CONFIG_FILEPATH
from src.hotel_reservations.pipelines.pip_07_prediction_pipeline import ConfigurationManager, PredictionPipeline
from src.hotel_reservations.pipelines.pip_04_model_trainer import TrainingPipeline
from src.hotel_reservations.exception import CustomException


# Initialize flask app
app = Flask(__name__)  # Specify the static folder

# Load configuration and prediction pipeline
try:
    config_manager = ConfigurationManager(DATA_TRANSFORMATION_CONFIG_FILEPATH)
    prediction_config = config_manager.get_prediction_pipeline_config()
    prediction_pipeline = PredictionPipeline(prediction_config)
except Exception as e:
    logger.error(f"Error initializing prediction pipeline: {e}")
    sys.exit(1)

# Define endpoints
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Execute training process
        logger.info("Starting training process")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_training()
        logger.info("Training process completed successfully")

        return jsonify({"message": "Model training triggered successfully"}), 200

    except CustomException as e:
        logger.error(f"Error during training: {e}")
        return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        logger.info(f"Received data for prediction: {data}")

        # Validate input data (add validation as needed)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert input data to pandas DataFrame
        input_df = pd.DataFrame([data])

        # Make predictions
        predictions = prediction_pipeline.make_predictions(input_df)
        predictions = [int(pred) for pred in predictions]

        logger.info(f"Prediction result: {predictions[0]}")
        return jsonify({"prediction": predictions[0]}), 200

    except CustomException as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)