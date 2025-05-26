# Predictive Maintenance: Turbofan Engine RUL Prediction

This project implements a machine learning pipeline to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. It supports multiple datasets from C-MAPSS (FD001-FD004), features a configurable pipeline, and allows for training, evaluation, and data exploration via a user-friendly interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Interactive Menu](#interactive-menu)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Modes of Operation](#modes-of-operation)
- [Methodology](#methodology)
- [Outputs](#outputs)
- [Future Work & Enhancements](#future-work--enhancements)
- [License](#license)

## Overview

Predictive maintenance is crucial for reducing operational costs and enhancing the safety of complex machinery like aircraft engines. This project aims to predict the RUL of turbofan engines by analyzing sensor data collected during their operation. A Random Forest Regressor model is trained on historical data from one or more of the C-MAPSS datasets (FD001, FD002, FD003, FD004) to make these predictions. The pipeline is designed to be configurable and extensible.

## Features

*   **Multi-Dataset Support:** Handles all four C-MAPSS datasets (FD001, FD002, FD003, FD004).
*   **Configurable Pipeline:** Key parameters (file paths, window sizes, sensors to remove/scale, RUL cap) are managed in a central configuration file (`config/config.py`).
*   **Data Loading & Preprocessing:** Loads and preprocesses turbofan engine sensor data for specified datasets.
*   **Irrelevant Sensor Removal:** Removes sensors identified as low-variance or highly correlated based on configurations in `config.py`.
*   **Feature Engineering:**
    *   Calculates rolling window statistics (mean, standard deviation) for sensor readings.
    *   Computes the Remaining Useful Life (RUL) for training data, with an option to cap the maximum RUL.
*   **Model Training:**
    *   Uses a `scikit-learn` pipeline.
    *   Applies `StandardScaler` to selected features (operational settings, original sensors, and their rolling statistics).
    *   Trains a `RandomForestRegressor` model.
    *   Saves trained models for each dataset to the `models/` directory.
*   **Model Evaluation:**
    *   Loads saved models.
    *   Prepares test data by taking the last cycle for each engine unit.
    *   Evaluates the model's performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
*   **Visualization:** Plots predicted RUL against actual RUL for evaluation.
*   **Data Exploration:** Includes functionality (`src/data_exploration.py`, accessible via `main.py`) for initial exploratory data analysis (EDA) on training data, generating:
    *   Dataset summaries and info.
    *   Correlation heatmaps.
    *   Lists of highly correlated feature pairs.
    *   Distribution plots for each sensor.
    *   EDA outputs are saved to `exploration_outputs/<dataset_name>/`.
*   **User Interface:** Offers both an interactive menu and command-line arguments for running different project tasks (train, evaluate, explore).

## Dataset

This project uses the **NASA Turbofan Engine Degradation Simulation Data Set (C-MAPSS)**.
The dataset consists of multivariate time series data from simulated turbofan engines under different operational conditions and fault modes.

*   **Subsets:** The full dataset includes four subsets (FD001, FD002, FD003, FD004), each with its own training and test data, and ground truth RUL for the test data. This project is configured to work with all four.
*   **Data Files Required (per subset, e.g., FD001):**
    *   `train_FD001.txt`: Training data.
    *   `test_FD001.txt`: Test data (trajectories end prior to failure).
    *   `RUL_FD001.txt`: Ground truth RUL values for the test data.

You can download the dataset from the NASA Prognostics Data Repository: [C-MAPSS Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) (look for "Damage Propagation Modeling" or similar zip file containing the text files).

## Project Structure


CMAPSS-PREDICTION/
├── config/
│ ├── init.py
│ └── config.py # Configuration file for paths, parameters, dataset-specific settings
├── data/ # (Gitignored) Store dataset files here (e.g., train_FD001.txt)
├── exploration_outputs/ # (Gitignored) Stores EDA plots, organized by dataset
├── models/ # (Gitignored) Stores trained .pkl model files, organized by dataset
├── src/
│ ├── init.py
│ ├── data_exploration.py # Functions for exploratory data analysis
│ ├── data_processing.py # Functions for loading and initial data cleaning
│ ├── evaluation.py # Functions for model evaluation
│ ├── feature_engineering.py# Functions for creating new features
│ ├── model.py # Function to build the ML pipeline
│ └── visualization.py # Functions for plotting results
├── .gitignore # Specifies intentionally untracked files
├── main.py # Main script to run training, evaluation, and exploration
├── README.md # This file
└── requirements.txt # Python dependencies

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   pip

2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd CMAPSS-PREDICTION
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the Dataset:**
    *   Download the C-MAPSS dataset (e.g., "Damage Propagation Modeling" zip file) from the link provided in the [Dataset](#dataset) section.
    *   Extract the files.
    *   Create a `data/` directory in the project root if it doesn't exist.
    *   Copy the relevant data files (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, and similarly for FD002, FD003, FD004) into the `data/` directory.
    *   *Ensure the file paths in `config/config.py` under the `DATASETS` dictionary match the location and names of your data files.*

6.  **Create Output Directories (if not automatically created by the script on first run):**
    The scripts will attempt to create these, but you can pre-create them:
    ```bash
    mkdir models
    mkdir exploration_outputs
    ```
    These directories are listed in `.gitignore`.

## Configuration

The primary configuration for the project is located in `config/config.py`. This file allows you to customize various aspects of the pipeline:

*   `WINDOW_SIZE`: Size of the rolling window for feature engineering.
*   `RUL_CAP`: Maximum value to cap the RUL in the training data.
*   `DATASETS`: A dictionary defining configurations for each C-MAPSS subset (FD001-FD004).
    *   `TRAIN_PATH`, `TEST_PATH`, `RUL_PATH`: Paths to the dataset files.
    *   `LOW_VARIANCE_SENSORS`: List of sensors to remove due to low variance (can be customized per dataset if needed, though currently shared).
    *   `HIGHLY_CORRELATED_SENSORS`: List of sensors to remove due to high correlation with other sensors.
*   `get_config(dataset_name)`: A helper function that consolidates settings for a specific dataset, determining `TO_REMOVE` (sensors to drop) and `COLUMNS_TO_SCALE` (features for `StandardScaler`).

## Usage

The project can be primarily interacted with via `main.py`, which offers an interactive menu or accepts command-line arguments.

### Interactive Menu

To use the interactive menu, run `main.py` without any arguments:

You will be presented with the following options:

Train Models: Select this to train models. You'll be prompted to enter comma-separated dataset names (e.g., FD001,FD002).

Evaluate Models: Select this to evaluate previously trained models. You'll be prompted for dataset names.

Run Exploratory Data Analysis: Select this to perform EDA on specified datasets.

Exit: Close the program.

Command-Line Interface (CLI)

You can also run specific tasks directly from the command line:

python main.py <mode> <dataset_name1> [<dataset_name2> ...]

<mode>: Specifies the operation to perform.

<dataset_nameX>: One or more dataset names (e.g., FD001, FD004).

Modes of Operation

train: Trains a model for each specified dataset and saves it to models/<dataset_name>_model.pkl.
evaluate: Loads a pre-trained model for each specified dataset, predicts RUL on its test set, and prints evaluation metrics (MSE, RMSE, R²), then displays a predicted vs. actual RUL plot.
explore: Runs exploratory data analysis for each specified dataset. This includes printing summary statistics, and generating/saving correlation heatmaps and sensor distribution plots to exploration_outputs/<dataset_name>/.


Note on Standalone EDA Script:
The EDA logic resides in src/data_exploration.py. While main.py provides a convenient way to trigger it, you can also run it directly (e.g., for development or specific analysis on one dataset) if you navigate to the src directory or adjust Python's path:

# From the project root directory
python -m src.data_exploration FD001


Methodology

Configuration Loading: For a given dataset (e.g., FD001), its specific configuration (file paths, sensors to remove, columns to scale) is loaded from config/config.py.

Data Loading: Training and test datasets are loaded using pandas. Column names are assigned based on the C-MAPSS dataset description.

Sensor Pruning: Irrelevant sensors (as defined by LOW_VARIANCE_SENSORS and HIGHLY_CORRELATED_SENSORS in config.py for the chosen dataset) are removed.

Feature Engineering:

Rolling Statistics: For selected sensor columns, rolling mean and standard deviation are calculated over a specified WINDOW_SIZE for each engine unit. These become new features.

RUL Calculation (Training Data): The RUL for each cycle in the training data is calculated as the difference between the maximum cycle for that unit and the current cycle. This RUL is then optionally capped at RUL_CAP.

Data Preparation for Model:

Training Set: Features include operational settings, selected original sensor readings, and their engineered rolling features. The target is the calculated RUL.

Test Set: For RUL prediction on the test set, data for each engine unit is processed similarly to the training data (sensor pruning, rolling features). The final instance (last recorded cycle) for each unit in the test set is used for prediction, as is standard for C-MAPSS evaluation.

Model Pipeline:

A ColumnTransformer applies StandardScaler to a defined list of columns (COLUMNS_TO_SCALE from config, which includes operational settings, kept original sensors, and their rolling features). Other columns are passed through.

A RandomForestRegressor is used as the prediction model.

Training: The pipeline is trained on the prepared training data. Trained models are saved to models/<dataset_name>_model.pkl.

Prediction & Evaluation:

The trained pipeline (loaded from file) predicts RUL for the prepared test data.

Predictions are compared against the true RUL values (from RUL_FDxxx.txt, also capped by RUL_CAP for consistency) using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

A scatter plot visualizes predicted vs. actual RUL.

Outputs

Trained Models: Saved as .pkl files in the models/ directory, named models/<dataset_name>_model.pkl.

Evaluation Metrics: MSE, RMSE, and R² scores are printed to the console during the evaluate mode.

Prediction Plots: A scatter plot of "Predicted vs Actual RUL" is displayed during the evaluate mode for each dataset.

Exploratory Data Analysis (EDA) Results:

Summary statistics printed to the console.

Plots (correlation heatmap, sensor distributions) saved as PNG files in exploration_outputs/<dataset_name>/.

Future Work & Enhancements

Generalize Configuration: Further refine config.py to allow more distinct configurations per dataset (FD001-FD004) if initial EDA suggests different sensors should be removed/scaled for optimal performance on each.

Hyperparameter Tuning: Implement techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameters for the RandomForestRegressor (or other models).

Advanced Models: Experiment with other regression models (e.g., Gradient Boosting, XGBoost, LightGBM) or neural networks (LSTMs, Transformers) suitable for time series data.

Cross-Validation: Implement robust cross-validation strategies suitable for time-series data, especially when generalizing to all datasets.

Dynamic Sensor Selection: Implement automated methods for selecting relevant sensors based on statistical properties or feature importance scores from models.

Error Analysis: Perform a more in-depth analysis of prediction errors to understand where the model performs poorly (e.g., for specific units, or at different stages of degradation).

Improved Test Set Handling: Explore using more of the test set history for prediction if applicable models (like LSTMs) are used, rather than just the last cycle.