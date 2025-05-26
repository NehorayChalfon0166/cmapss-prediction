# Constants and imports
from config.config import DATASETS, get_config, WINDOW_SIZE, RUL_CAP
from src.data_processing import load_data, remove_irrelevant_sensors
from src.feature_engineering import add_rolling_features, calculate_rul
from src.model import build_pipeline
from src.evaluation import evaluate
from src.visualization import plot_results
from src.data_exploration import explore_dataset
import pandas as pd
import pickle
import os
import sys

MODEL_DIR = "models"
MODE_TRAIN = "train"
MODE_EVALUATE = "evaluate"
MODE_EXPLORE = "explore"
VALID_MODES = [MODE_TRAIN, MODE_EVALUATE, MODE_EXPLORE]


# === Core functionality ===

def train_model(dataset_name):
    """Train and evaluate the model for a specific dataset."""
    config = get_config(dataset_name)
    print(f"\n=== Training Model for Dataset: {dataset_name} ===")
    train = load_data(config["TRAIN_PATH"])
    train = remove_irrelevant_sensors(train, config["TO_REMOVE"])
    sensor_cols = [col for col in config["COLUMNS_TO_SCALE"] if 'sensor' in col]
    train = add_rolling_features(train, sensor_cols, WINDOW_SIZE)
    train = calculate_rul(train, cap=RUL_CAP)

    x_train = train.drop(columns=['unit_id', 'cycle', 'RUL'])
    y_train = train['RUL']

    pipeline = build_pipeline(config["COLUMNS_TO_SCALE"], WINDOW_SIZE)
    pipeline.fit(x_train, y_train)

    save_model(pipeline, dataset_name)
    print(f"Model for {dataset_name} has been trained and saved.")

def evaluate_model(dataset_name):
    """Evaluate a saved model against true RUL values."""
    config = get_config(dataset_name)
    pipeline = load_model(dataset_name)
    if pipeline is None:
        return

    print(f"\n=== Evaluating Model for Dataset: {dataset_name} ===")
    x_test = prepare_test_set(config)
    y_pred = pipeline.predict(x_test)

    y_true = pd.read_csv(config["RUL_PATH"], sep='\s+', header=None, names=['RUL'])['RUL'].clip(upper=RUL_CAP)

    if len(y_pred) != len(y_true):
        print("Mismatch in evaluation: predicted RUL and true RUL lengths do not match.")
        return

    evaluate(y_true, y_pred)
    plot_results(y_true, y_pred, title=f"Predicted vs Actual RUL ({dataset_name})")

def run_exploratory_analysis(dataset_name):
    """Run exploratory data analysis for a specific dataset."""
    print(f"\n=== Running Exploratory Data Analysis for Dataset: {dataset_name} ===")
    explore_dataset(dataset_name)


# === Helper functions ===

def prepare_test_set(config):
    """Prepare the test set for evaluation or prediction."""
    test = load_data(config["TEST_PATH"])
    test = remove_irrelevant_sensors(test, config["TO_REMOVE"])
    sensor_cols = [col for col in config["COLUMNS_TO_SCALE"] if 'sensor' in col]
    test = add_rolling_features(test, sensor_cols, WINDOW_SIZE)
    test_last = test.groupby('unit_id').last().drop(columns=['cycle'], errors='ignore')
    return test_last

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

def process_datasets(dataset_names, mode):
    """Process datasets based on the selected mode."""
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        if dataset_name in DATASETS:
            if mode == MODE_TRAIN:
                train_model(dataset_name)
            elif mode == MODE_EVALUATE:
                evaluate_model(dataset_name)
            elif mode == MODE_EXPLORE:
                run_exploratory_analysis(dataset_name)
        else:
            print(f"Invalid dataset name: {dataset_name}")


# === User interaction ===

def display_menu():
    """Display the main menu and handle user input."""
    while True:
        print("\n=== Main Menu ===")
        print("1. Train Models")
        print("2. Evaluate Models")
        print("3. Run Exploratory Data Analysis")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("\nAvailable datasets for training:")
            print(", ".join(DATASETS.keys()))
            dataset_names = input("Enter the dataset names to train on (comma-separated): ").split(",")
            process_datasets(dataset_names, MODE_TRAIN)

        elif choice == "2":
            print("\nAvailable datasets for evaluation:")
            print(", ".join(DATASETS.keys()))
            dataset_names = input("Enter the dataset names to evaluate (comma-separated): ").split(",")
            process_datasets(dataset_names, MODE_EVALUATE)

        elif choice == "3":
            print("\nAvailable datasets for exploratory data analysis:")
            print(", ".join(DATASETS.keys()))
            dataset_names = input("Enter the dataset names to analyze (comma-separated): ").split(",")
            process_datasets(dataset_names, MODE_EXPLORE)

        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def handle_cli_arguments(args):
    """Handle command-line arguments for the script."""
    if len(args) == 2:
        print("Error: No dataset names provided. Please specify at least one dataset.")
        print("Usage: python main.py <mode> <dataset_name1> <dataset_name2> ...")
        print(f"Modes: {', '.join(VALID_MODES)}")
    elif len(args) > 2:
        mode = args[1]
        dataset_names = args[2:]
        if mode not in VALID_MODES:
            print(f"Invalid mode. Use one of: {', '.join(VALID_MODES)}.")
            print("Usage: python main.py <mode> <dataset_name1> <dataset_name2> ...")
        else:
            process_datasets(dataset_names, mode)


# === Entry point ===

if __name__ == "__main__":
    if len(sys.argv) > 1:
        handle_cli_arguments(sys.argv)
    else:
        display_menu()
