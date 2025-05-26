# filepath: main.py
from config.config import *
from src.data_processing import load_data, remove_irrelevant_sensors
from src.feature_engineering import add_rolling_features, calculate_rul
from src.model import build_pipeline
from src.evaluation import evaluate
from src.visualization import plot_results
import pandas as pd
import sys

def main(dataset_name):
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <dataset_name>")
        print("Example: python main.py FD001")
    else:
        dataset_name = sys.argv[1]
        main(dataset_name)