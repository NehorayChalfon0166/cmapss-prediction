# Predictive Maintenance: Turbofan Engine RUL Prediction

This project implements a machine learning pipeline to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The current implementation primarily focuses on the FD001 subset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Data Exploration](#data-exploration)
  - [Training and Prediction](#training-and-prediction)
  - [Configuration](#configuration)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work & Enhancements](#future-work--enhancements)
- [License](#license)

## Overview

Predictive maintenance is crucial for reducing operational costs and enhancing the safety of complex machinery like aircraft engines. This project aims to predict the RUL of turbofan engines by analyzing sensor data collected during their operation. A Random Forest Regressor model is trained on historical data to make these predictions.

## Features

*   **Data Loading & Preprocessing:** Loads and preprocesses the turbofan engine sensor data.
*   **Irrelevant Sensor Removal:** Removes sensors identified as low-variance or highly correlated based on initial analysis (configurable).
*   **Feature Engineering:**
    *   Calculates rolling window statistics (mean, standard deviation) for sensor readings.
    *   Computes the Remaining Useful Life (RUL) for training data, with an option to cap the maximum RUL.
*   **Model Training:**
    *   Uses a `scikit-learn` pipeline.
    *   Applies `StandardScaler` to selected features.
    *   Trains a `RandomForestRegressor` model.
*   **Evaluation:** Evaluates the model's performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
*   **Visualization:** Plots predicted RUL against actual RUL.
*   **Data Exploration:** Includes a separate script (`data_exploration.py`) for initial exploratory data analysis (EDA) on the training data, generating summaries and visualizations.
*   **Configurable Pipeline:** Key parameters (file paths, window sizes, sensors to remove/scale, RUL cap) are managed in a central configuration file (`config/config.py`).

## Dataset

This project uses the **NASA Turbofan Engine Degradation Simulation Data Set (C-MAPSS)**.
The dataset consists of multivariate time series data from simulated turbofan engines under different operational conditions and fault modes.

*   **Subsets:** The full dataset includes four subsets (FD001, FD002, FD003, FD004), each with its own training and test data. The test data trajectories end some time prior to system failure.
*   **Current Implementation:** This project's `main.py` and `config.py` are primarily set up for the **FD001** subset.
*   **Data Files for FD001:**
    *   `train_FD001.txt`: Training data.
    *   `test_FD001.txt`: Test data.
    *   `RUL_FD001.txt`: Ground truth RUL values for the test data.

You can download the dataset from the NASA Prognostics Data Repository: [C-MAPSS Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)


## Project Structure

.
├── config/
│ └── config.py # Configuration file for paths, parameters
├── data/ # To store dataset files
├── exploration_outputs/ # Created by data_exploration.py and Stores EDA plots
├── src/
│ ├── data_processing.py # Functions for loading and initial data cleaning
│ ├── evaluation.py # Functions for model evaluation
│ ├── feature_engineering.py# Functions for creating new features
│ ├── model.py # Function to build the ML pipeline
│ └── visualization.py # Functions for plotting results
├── data_exploration.py # Script for exploratory data analysis
├── main.py # Main script to run the training and prediction pipeline
├── requirements.txt # Python dependencies
└── README.md # This file

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   pip

2.  **Clone the repository:**
    git clone <your-repository-url>
    cd <repository-name>

3.  **Create and activate a virtual environment (recommended):**
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

4.  **Install dependencies:**
    pip install -r requirements.txt

5.  **Download the Dataset:**
    *   Download the C-MAPSS dataset (e.g., "Damage Propagation Modeling" zip file) from the link provided in the [Dataset](#dataset) section.
    *   Extract the files.
    *   Create a `data/` directory in the project root.
    *   Copy the relevant data files (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) into the `data/` directory.
        *Ensure the file paths in `config/config.py` match the location and names of your data files.*

## Usage

### Data Exploration

The `data_exploration.py` script can be used to perform an initial analysis of the training data.

This script will:

Print summary statistics and info about the dataset.

Generate and save a correlation heatmap to exploration_outputs/correlation_heatmap.png.

Identify and print highly correlated feature pairs.

Generate and save distribution plots for each sensor to the exploration_outputs/ directory.

## Training and Prediction

The main pipeline for training the RUL prediction model and evaluating it on the test set is run using main.py.

This script will:

Load and preprocess the training data.

Perform feature engineering (rolling features, RUL calculation).

Train the RandomForestRegressor model.

Load and preprocess the test data.

Make RUL predictions on the test data.

Load the true RUL values.

Evaluate the predictions and print MSE, RMSE, and R² scores.

Display a scatter plot of predicted vs. actual RUL.

## Configuration

Most parameters of the pipeline can be adjusted in `config/config.py`:

WINDOW_SIZE: Size of the rolling window for feature engineering.

RUL_CAP: Maximum value to cap the RUL.

TRAIN_PATH, TEST_PATH, RUL_PATH: Paths to the dataset files.

TO_REMOVE: List of sensors to remove before feature engineering and scaling.

COLUMNS_TO_SCALE: List of columns (original sensors and operational settings) to be scaled.

## Methodology

1. Data Loading: Training and test datasets are loaded. Column names are assigned as per the dataset description.

2. Sensor Pruning: Irrelevant sensors (low variance or highly correlated, identified via EDA and configured in config.py) are removed.

3. Feature Engineering:

    * Rolling Statistics: For each relevant sensor, rolling mean and standard deviation are calculated over a specified window for each unit.

    * RUL Calculation (Training Data): The RUL for each cycle in the training data is calculated as the difference between the maximum cycle for that unit and the current cycle. This RUL is then capped at a predefined value (RUL_CAP).

4. Data Splitting:

    * Training set: Features (sensor readings, operational settings, engineered rolling features) and the target RUL.

    * Test set: The last cycle of data for each unit is used for prediction, as per typical C-MAPSS evaluation.

5. Model Pipeline:

    * A ColumnTransformer applies StandardScaler only to specified sensor columns and their derived rolling features.

    * A RandomForestRegressor is used as the prediction model.

6. Prediction: The trained pipeline predicts RUL for the prepared test data.

7. Evaluation: Predictions are compared against the true RUL values using MSE, RMSE, and R² score. A scatter plot visualizes the prediction accuracy.

## Results

MSE = XXX.XX
RMSE = XX.XX
R² score = 0.XXXX
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

A plot showing "Predicted vs Actual RUL" will also be displayed.

## Future Work & Enhancements

Generalize for all C-MAPSS Datasets (FD001-FD004):

Modify config.py and main.py to easily switch between or iterate over the FD001, FD002, FD003, and FD004 datasets. This will likely involve different configurations for TO_REMOVE and COLUMNS_TO_SCALE for each dataset.

Update data_exploration.py to accept dataset choice as an argument.

Hyperparameter Tuning: Implement techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameters for the RandomForestRegressor (or other models).

Advanced Models: Experiment with other regression models (e.g., Gradient Boosting, XGBoost, LightGBM, Neural Networks like LSTMs or Transformers suited for time series).

Cross-Validation: Implement robust cross-validation strategies suitable for time-series data, especially when generalizing to all datasets.

Dynamic Sensor Selection: Implement automated methods for selecting relevant sensors based on statistical properties or feature importance scores.

Error Analysis: Perform a more in-depth analysis of prediction errors to understand where the model performs poorly.

Saving and Loading Models: Add functionality to save trained models and load them for later use.