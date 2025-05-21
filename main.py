import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# ----------------------------------------
# Configuration
# ----------------------------------------

# File paths
train_path = 'train_FD001.txt'
test_path = 'test_FD001.txt'
rul_path = 'RUL_FD001.txt'

WINDOW_SIZE = 10
RUL_CAP = 125

# Sensors identified for removal
LOW_VARIANCE_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_8', 'sensor_13', 'sensor_9', 'sensor_21']
TO_REMOVE = LOW_VARIANCE_SENSORS + HIGHLY_CORRELATED_SENSORS

# Sensors for scaling
COLUMNS_TO_SCALE = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
]

# ----------------------------------------
# Functions
# ----------------------------------------

def load_data(path: str) -> pd.DataFrame:
    # Define the expected column names
    columns = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    return pd.read_csv(path, sep='\s+', header=None, names=columns)


def remove_irrelevant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=TO_REMOVE)


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int) -> pd.DataFrame:
    for sensor in sensors:
        rolling_mean = df.groupby('unit_id')[sensor].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_std = df.groupby('unit_id')[sensor].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        df[f'{sensor}_rolling_mean_{window}'] = rolling_mean
        df[f'{sensor}_rolling_std_{window}'] = rolling_std
    return df


def calculate_rul(df: pd.DataFrame, cap_value: int = None) -> pd.DataFrame:
    max_cycle = df.groupby('unit_id')['cycle'].max()
    df['RUL'] = df['unit_id'].map(max_cycle) - df['cycle']
    if cap_value is not None:
        df['RUL'] = df['RUL'].clip(upper=cap_value)
    return df


def sequential_split(df: pd.DataFrame, test_size=0.2):
      train_list = []
      test_list = []

      # Group by unit_id
      grouped = df.groupby('unit_id')

      for unit_id, group in grouped:
            cycle_split = group['cycle'].quantile(1-test_size)
            # Split the data into training and testing sets
            train_data = group[group['cycle'] <= cycle_split]
            test_data = group[group['cycle'] > cycle_split]
            train_list.append(train_data)
            test_list.append(test_data)

      # Concatenate all unit_id groups to form the final train and test DataFrames
      train_df = pd.concat(train_list, axis=0)
      test_df = pd.concat(test_list, axis=0)

      return train_df, test_df


def build_pipeline(columns_to_scale: list) -> Pipeline:
    extended_columns = columns_to_scale.copy()
    for col in columns_to_scale:
        if 'sensor' in col:
            extended_columns += [f"{col}_rolling_mean_{WINDOW_SIZE}", f"{col}_rolling_std_{WINDOW_SIZE}"]
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), extended_columns)],
        remainder='passthrough'
    )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE = {mse:.2f}')
    print(f'RMSE = {rmse:.2f}') # RMSE is often more interpretable
    print(f'R² score = {r2:.4f}')
    return mse, r2, rmse


def plot_results(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs Actual RUL")
    plt.grid(True)
    plt.show()


# ----------------------------------------
# Main pipeline
# ----------------------------------------

def main():
    # Load and preprocess data
    train_df = load_data(train_path)
    train_df = remove_irrelevant_sensors(train_df)
    sensor_cols = [col for col in train_df.columns if 'sensor_' in col]
    df_train_processed  = add_rolling_features(train_df, sensor_cols, WINDOW_SIZE)
    df_train_processed  = calculate_rul(df_train_processed, cap_value=RUL_CAP)

    # Separate features and target
    x_train = df_train_processed.drop(columns=['unit_id', 'cycle', 'RUL'])
    y_train = df_train_processed['RUL']

    # Build and train pipeline
    pipeline = build_pipeline(COLUMNS_TO_SCALE)
    pipeline.fit(x_train, y_train)
    
    test_df = load_data(test_path)  # Load all matching datasets
    train_df = remove_irrelevant_sensors(train_df)
    df_train_processed  = add_rolling_features(train_df, sensor_cols, WINDOW_SIZE)
    x_test = 
    # Evaluate and visualize
    print("\nTraining set RUL statistics:")
    print(y_train.describe())
    print("\nTest set RUL statistics:")
    print(y_test.describe())
    evaluate(y_test, y_pred)
    plot_results(y_test, y_pred)

# ----------------------------------------
# Run
# ----------------------------------------

if __name__ == '__main__':
    main()
