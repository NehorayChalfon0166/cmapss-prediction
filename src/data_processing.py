# filepath: src/data_processing.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    columns = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    return pd.read_csv(path, sep='\s+', header=None, names=columns)

def remove_irrelevant_sensors(df: pd.DataFrame, to_remove: list) -> pd.DataFrame:
    return df.drop(columns=[col for col in to_remove if col in df.columns])