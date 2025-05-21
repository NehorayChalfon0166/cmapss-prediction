# filepath: src/model.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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