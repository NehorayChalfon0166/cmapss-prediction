# filepath: src/evaluation.py
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE = {mse:.2f}')
    print(f'RMSE = {rmse:.2f}')
    print(f'R² score = {r2:.4f}')
    return mse, r2, rmse