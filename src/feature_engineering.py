# filepath: src/feature_engineering.py
import pandas as pd

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