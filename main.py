import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIG ===
WINDOW_SIZE = 10
RUL_CAP = 125
TRAIN_PATH = 'train_FD001.txt'
TEST_PATH = 'test_FD001.txt'
RUL_PATH = 'RUL_FD001.txt'

LOW_VARIANCE_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21']
TO_REMOVE = sorted(set(LOW_VARIANCE_SENSORS + HIGHLY_CORRELATED_SENSORS))

COLUMNS_TO_SCALE = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
]
COLUMNS_TO_SCALE = [col for col in COLUMNS_TO_SCALE if col not in TO_REMOVE]

# === FUNCTIONS ===
def load_data(path: str) -> pd.DataFrame:
    columns = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    return pd.read_csv(path, sep='\s+', header=None, names=columns)

def remove_irrelevant_sensors(df: pd.DataFrame, to_remove: list) -> pd.DataFrame:
    return df.drop(columns=[col for col in to_remove if col in df.columns])

def add_rolling_features(df: pd.DataFrame, features: list, window: int) -> pd.DataFrame:
    df_out = df.copy()
    for f in features:
        if f in df_out.columns:
            grouped = df_out.groupby('unit_id')[f]
            df_out[f'{f}_rolling_mean_{window}'] = grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df_out[f'{f}_rolling_std_{window}'] = grouped.rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    return df_out

def calculate_rul(df: pd.DataFrame, cap: int = None) -> pd.DataFrame:
    df_out = df.copy()
    max_cycle = df_out.groupby('unit_id')['cycle'].max()
    df_out['RUL'] = df_out['unit_id'].map(max_cycle) - df_out['cycle']
    if cap is not None:
        df_out['RUL'] = df_out['RUL'].clip(upper=cap)
    return df_out

def build_pipeline(columns: list, window_size: int) -> Pipeline:
    extended = []
    for col in columns:
        extended.append(col)
        if 'sensor' in col:
            extended.append(f"{col}_rolling_mean_{window_size}")
            extended.append(f"{col}_rolling_std_{window_size}")
    extended = sorted(set(extended))
    transformer = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), extended)],
        remainder='passthrough'
    )

    return Pipeline([
        ('preprocessing', transformer),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

def evaluate(y_true: pd.Series, y_pred: np.ndarray):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE = {mse:.2f}')
    print(f'RMSE = {rmse:.2f}')
    print(f'R² score = {r2:.4f}')
    return mse, r2, rmse

def plot_results(y_true: pd.Series, y_pred: np.ndarray, title="Predicted vs Actual RUL"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.grid(True)
    plt.show()

# === MAIN EXECUTION ===
def main():
    print("=== Training Phase ===")
    train = load_data(TRAIN_PATH)
    train = remove_irrelevant_sensors(train, TO_REMOVE)
    # Apply rolling features only to sensor columns
    sensor_cols = [col for col in COLUMNS_TO_SCALE if 'sensor' in col]
    train = add_rolling_features(train, sensor_cols, WINDOW_SIZE)
    train = calculate_rul(train, cap=RUL_CAP)

    x_train = train.drop(columns=['unit_id', 'cycle', 'RUL'])
    y_train = train['RUL']

    pipeline = build_pipeline(COLUMNS_TO_SCALE, WINDOW_SIZE)
    pipeline.fit(x_train, y_train)

    print("=== Test Phase ===")
    test = load_data(TEST_PATH)
    test = remove_irrelevant_sensors(test, TO_REMOVE)
    # Apply rolling features only to sensor columns
    sensor_cols = [col for col in COLUMNS_TO_SCALE if 'sensor' in col]
    test = add_rolling_features(test, sensor_cols, WINDOW_SIZE)
    test_last = test.groupby('unit_id').last().drop(columns=['cycle'], errors='ignore')
    x_test = test_last[x_train.columns]

    y_pred = pipeline.predict(x_test)

    print("=== Evaluation Phase ===")
    y_true = pd.read_csv(RUL_PATH, sep='\s+', header=None, names=['RUL'])['RUL'].clip(upper=RUL_CAP)

    if len(y_pred) != len(y_true):
        print("Mismatch in prediction and true RUL lengths")
        return

    evaluate(y_true, y_pred)
    plot_results(y_true, y_pred, title="Predicted vs Actual RUL (FD001)")

if __name__ == "__main__":
    main()
