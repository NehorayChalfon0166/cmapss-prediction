# filepath: main.py
from config.config import *
from src.data_processing import load_data, remove_irrelevant_sensors
from src.feature_engineering import add_rolling_features, calculate_rul
from src.model import build_pipeline
from src.evaluation import evaluate
from src.visualization import plot_results
import pandas as pd
import pickle
import os
import sys

MODEL_DIR = "models"

def save_model(model, dataset_name):
    """Save the trained model to a file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(dataset_name):
    """Load a trained model from a file."""
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.pkl")
    if not os.path.exists(model_path):
        print(f"No saved model found for {dataset_name}. Please train the model first.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def train_and_evaluate(dataset_name):
    """Train and evaluate the model for a specific dataset."""
    config = get_config(dataset_name)
    if not config:
        print(f"Dataset {dataset_name} not found in configuration.")
        return

    print(f"=== Training Phase for {dataset_name} ===")
    train = load_data(config["TRAIN_PATH"])
    train = remove_irrelevant_sensors(train, config["TO_REMOVE"])
    sensor_cols = [col for col in config["COLUMNS_TO_SCALE"] if 'sensor' in col]
    train = add_rolling_features(train, sensor_cols, WINDOW_SIZE)
    train = calculate_rul(train, cap=RUL_CAP)

    x_train = train.drop(columns=['unit_id', 'cycle', 'RUL'])
    y_train = train['RUL']

    pipeline = build_pipeline(config["COLUMNS_TO_SCALE"], WINDOW_SIZE)
    pipeline.fit(x_train, y_train)

    # Save the trained model
    save_model(pipeline, dataset_name)

    print(f"=== Test Phase for {dataset_name} ===")
    test = load_data(config["TEST_PATH"])
    test = remove_irrelevant_sensors(test, config["TO_REMOVE"])
    test = add_rolling_features(test, sensor_cols, WINDOW_SIZE)
    test_last = test.groupby('unit_id').last().drop(columns=['cycle'], errors='ignore')
    x_test = test_last[x_train.columns]

    y_pred = pipeline.predict(x_test)

    print(f"=== Evaluation Phase for {dataset_name} ===")
    y_true = pd.read_csv(config["RUL_PATH"], sep='\s+', header=None, names=['RUL'])['RUL'].clip(upper=RUL_CAP)

    if len(y_pred) != len(y_true):
        print("Mismatch in prediction and true RUL lengths")
        return

    evaluate(y_true, y_pred)
    plot_results(y_true, y_pred, title=f"Predicted vs Actual RUL ({dataset_name})")

def predict_with_saved_model(dataset_name):
    """Make predictions using a saved model."""
    config = get_config(dataset_name)
    if not config:
        print(f"Dataset {dataset_name} not found in configuration.")
        return

    # Load the saved model
    pipeline = load_model(dataset_name)
    if not pipeline:
        return

    print(f"=== Prediction Phase for {dataset_name} ===")
    test = load_data(config["TEST_PATH"])
    test = remove_irrelevant_sensors(test, config["TO_REMOVE"])
    sensor_cols = [col for col in config["COLUMNS_TO_SCALE"] if 'sensor' in col]
    test = add_rolling_features(test, sensor_cols, WINDOW_SIZE)
    test_last = test.groupby('unit_id').last().drop(columns=['cycle'], errors='ignore')
    x_test = test_last

    y_pred = pipeline.predict(x_test)
    print(f"Predictions for {dataset_name}:")
    print(y_pred)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <mode> <dataset_name1> <dataset_name2> ...")
        print("Modes: train, predict")
        print("Example: python main.py train FD001 FD002")
        print("Example: python main.py predict FD001")
    else:
        mode = sys.argv[1]
        dataset_names = sys.argv[2:]

        for dataset_name in dataset_names:
            print(f"\n=== Processing {dataset_name} ===")
            if mode == "train":
                train_and_evaluate(dataset_name)
            elif mode == "predict":
                predict_with_saved_model(dataset_name)
            else:
                print(f"Invalid mode: {mode}. Use 'train' or 'predict'.")