import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# train_test_split is not used in this version of main
import matplotlib.pyplot as plt
import numpy as np
# ----------------------------------------
# Configuration
# ----------------------------------------

# File paths
train_path = 'train_FD001.txt'
test_path = 'test_FD001.txt'
rul_path = 'RUL_FD001.txt' # Make sure this filename is correct

WINDOW_SIZE = 10
RUL_CAP = 125

# Sensors identified for removal
LOW_VARIANCE_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
# Ensure your HIGHLY_CORRELATED_SENSORS list is what you intend.
# Example: if sensor_8 and sensor_13 are correlated, pick one to remove.
# Current list removes both sensor_8 and sensor_13, sensor_9 (for sensor_14), sensor_11 & sensor_12 (for sensor_4/7)
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21'] # Keeping sensor_8, sensor_14

# Combine and ensure uniqueness
TO_REMOVE = sorted(list(set(LOW_VARIANCE_SENSORS + HIGHLY_CORRELATED_SENSORS)))

# Define original columns for context
operational_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_measurements_all = [f'sensor_{i}' for i in range(1, 22)]
all_feature_columns = operational_settings + sensor_measurements_all

# Define features to keep, which will be the base for scaling and rolling
FEATURES_TO_KEEP = [col for col in all_feature_columns if col not in TO_REMOVE]

# COLUMNS_TO_SCALE should list the base features whose values (and their rolling features if applicable) will be scaled.
# Based on your original list, it seems you intend to use a specific subset.
# Let's ensure this list only contains features that remain after TO_REMOVE.
COLUMNS_TO_SCALE = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', # Added sensor_8 as per above decision
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
]
# Filter COLUMNS_TO_SCALE to only include features not in TO_REMOVE
COLUMNS_TO_SCALE = [col for col in COLUMNS_TO_SCALE if col not in TO_REMOVE]

# ----------------------------------------
# Functions
# ----------------------------------------

def load_data(path: str) -> pd.DataFrame:
    columns = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    return pd.read_csv(path, sep='\s+', header=None, names=columns)


def remove_irrelevant_sensors(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    # Only drop columns that actually exist in the DataFrame
    columns_present_to_remove = [col for col in columns_to_remove if col in df.columns]
    return df.drop(columns=columns_present_to_remove)


def add_rolling_features(df: pd.DataFrame, features_to_roll: list, window: int) -> pd.DataFrame:
    df_out = df.copy()
    for feature_name in features_to_roll:
        # Ensure feature_name exists in df_out before trying to roll
        if feature_name in df_out.columns:
            rolling_mean = df_out.groupby('unit_id')[feature_name].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_std = df_out.groupby('unit_id')[feature_name].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            df_out[f'{feature_name}_rolling_mean_{window}'] = rolling_mean
            df_out[f'{feature_name}_rolling_std_{window}'] = rolling_std
        else:
            print(f"Warning: Column {feature_name} not found for rolling features.")
    return df_out


def calculate_rul(df: pd.DataFrame, cap_value: int = None) -> pd.DataFrame:
    df_out = df.copy()
    max_cycle = df_out.groupby('unit_id')['cycle'].max().to_dict() # Convert to dict for faster mapping
    df_out['RUL'] = df_out['unit_id'].map(max_cycle) - df_out['cycle']
    if cap_value is not None:
        df_out['RUL'] = df_out['RUL'].clip(upper=cap_value)
    return df_out

# sequential_split is not used in this main flow

def build_pipeline(columns_to_scale_base: list, window_size: int) -> Pipeline:
    extended_columns_for_scaling = []
    for col in columns_to_scale_base:
        extended_columns_for_scaling.append(col) # Add base feature
        # Add rolling features for sensors only, as per original logic
        if 'sensor' in col:
            extended_columns_for_scaling.append(f"{col}_rolling_mean_{window_size}")
            extended_columns_for_scaling.append(f"{col}_rolling_std_{window_size}")
    
    # Ensure no duplicates if a column name accidentally matches a rolled feature name format
    extended_columns_for_scaling = sorted(list(set(extended_columns_for_scaling)))

    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), extended_columns_for_scaling)],
        remainder='passthrough' 
    )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE = {mse:.2f}')
    print(f'RMSE = {rmse:.2f}')
    print(f'R² score = {r2:.4f}')
    return mse, r2, rmse


def plot_results(y_true, y_pred, title_suffix=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) # Diagonal line
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Predicted vs Actual RUL {title_suffix}")
    plt.grid(True)
    plt.show()


# ----------------------------------------
# Main pipeline
# ----------------------------------------

def main():
    print("--- Training Phase ---")
    df_train = load_data(train_path)
    df_train_processed = remove_irrelevant_sensors(df_train.copy(), TO_REMOVE)

    # Features for rolling are those in COLUMNS_TO_SCALE that are sensors, plus op_settings if desired
    # Your original `build_pipeline` adds rolling features only for 'sensor' columns in COLUMNS_TO_SCALE.
    # So, `add_rolling_features` should primarily target those same sensor columns.
    # Let's define features_to_roll based on COLUMNS_TO_SCALE to be consistent.
    features_to_roll_in_training = [col for col in COLUMNS_TO_SCALE] # Rolling all columns_to_scale
    
    df_train_processed = add_rolling_features(df_train_processed, features_to_roll_in_training, WINDOW_SIZE)
    df_train_processed = calculate_rul(df_train_processed, cap_value=RUL_CAP)

    x_train = df_train_processed.drop(columns=['unit_id', 'cycle', 'RUL'])
    y_train = df_train_processed['RUL']
    
    print(f"Training features: {x_train.columns.tolist()}")

    pipeline = build_pipeline(COLUMNS_TO_SCALE, WINDOW_SIZE) # Pass base columns for scaling
    pipeline.fit(x_train, y_train)
    
    print("\n--- Test Phase ---")
    df_test = load_data(test_path)
    df_test_processed = remove_irrelevant_sensors(df_test.copy(), TO_REMOVE)
    
    # Use the same features_to_roll_in_training for consistency
    df_test_processed = add_rolling_features(df_test_processed, features_to_roll_in_training, WINDOW_SIZE)

    # For the test set, we predict on the last cycle for each unit
    # Features for these last cycles are needed
    x_test_last_cycle_features = df_test_processed.groupby('unit_id').last()
    
    # Drop 'cycle' if it's still there (it's an index after groupby.last() if not reset)
    # and any other columns not in x_train.columns
    # Ensure x_test_last_cycle_features has the same columns as x_train
    x_test_final_features = x_test_last_cycle_features.drop(columns=['cycle'], errors='ignore')
    x_test_final_features = x_test_final_features[x_train.columns] # Align columns with training set

    print(f"Test features (after alignment): {x_test_final_features.columns.tolist()}")
    y_pred = pipeline.predict(x_test_final_features)

    # --- Evaluation Phase ---
    print("\n--- Evaluation on Official Test Set ---")
    try:
        y_true_test = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])['RUL']
    except FileNotFoundError:
        print(f"Error: Truth RUL file not found at {rul_path}")
        return
    
    y_true_test_capped = y_true_test.clip(upper=RUL_CAP)

    if len(y_pred) != len(y_true_test_capped):
        print("Error: Mismatch between number of predictions and true RUL values!")
        print(f"  Predictions: {len(y_pred)}, True RULs: {len(y_true_test_capped)}")
        return

    print("\nTest set (true, capped) RUL statistics:")
    print(y_true_test_capped.describe())
    print("\nPredicted RUL statistics (Official Test Set):")
    print(pd.Series(y_pred).describe())
    
    evaluate(y_true_test_capped, y_pred)
    plot_results(y_true_test_capped, y_pred, title_suffix="(Official Test Set FD001)")

# ----------------------------------------
# Run
# ----------------------------------------

if __name__ == '__main__':
    # Refined HIGHLY_CORRELATED_SENSORS (example: keep one from each pair)
    # sensor_4 vs sensor_11 (0.83) -> keep sensor_4, drop sensor_11
    # sensor_4 vs sensor_12 (-0.82) -> keep sensor_4, drop sensor_12
    # sensor_8 vs sensor_13 (0.83) -> keep sensor_8, drop sensor_13
    # sensor_9 vs sensor_14 (0.96) -> keep sensor_14, drop sensor_9
    # sensor_20 vs sensor_21 (your finding) -> keep sensor_20, drop sensor_21
    
    # Based on this, an updated HIGHLY_CORRELATED_SENSORS would be:
    HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21']
    TO_REMOVE = sorted(list(set(LOW_VARIANCE_SENSORS + HIGHLY_CORRELATED_SENSORS)))

    # COLUMNS_TO_SCALE should be the features you want to use, that are *not* in TO_REMOVE
    # This means your original COLUMNS_TO_SCALE needs to be consistent with TO_REMOVE.
    # Let's derive it from FEATURES_TO_KEEP if you want to scale all kept features.
    # Or, use your specific list if you only want to scale a subset of kept features.
    
    # Your original list:
    # COLUMNS_TO_SCALE_original = [
    #     'op_setting_1', 'op_setting_2', 'op_setting_3',
    #     'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    #     'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
    # ]
    # Filter this list by what's NOT in the updated TO_REMOVE
    COLUMNS_TO_SCALE = [col for col in COLUMNS_TO_SCALE if col not in TO_REMOVE]
    # This line ensures that the global COLUMNS_TO_SCALE used by build_pipeline is correctly set up.
    
    print(f"Final columns to remove: {TO_REMOVE}")
    print(f"Final columns for base scaling/rolling: {COLUMNS_TO_SCALE}")

    main()