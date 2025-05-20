import pandas
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pandas.read_csv('train_FD001.txt', sep='\s+', header=None,
                    names=['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                          'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                          'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
                          'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                          'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                          'sensor_21'])


### feature extraction
# from data exloration i saw that sensors 1, 5, 6, 10, 16, 18, 19
# are basiccly constants so have little to no effect
# and 'sensor_11','sensor_12','sensor_8','sensor_13','sensor_9' have strong correlations and can be removed
low_variance = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
            'sensor_16','sensor_18','sensor_19']

strong_correlation = ['sensor_11','sensor_12','sensor_8','sensor_13','sensor_9', 'sensor_21']

to_remove = low_variance + strong_correlation
df_reduced = df.drop(to_remove, axis=1)


# feature engineering
# Sensors to calculate rolling statistics for
selected_sensors = [column for column in df_reduced if 'sensor' in column]
window_size = 10
# Group by unit_id and calculate rolling statistics
rolling_features = []
for sensor in selected_sensors:
    rolling_mean = df_reduced.groupby('unit_id')[sensor].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_std = df_reduced.groupby('unit_id')[sensor].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    # Replace NaN values in rolling_std with 0
    rolling_std = rolling_std.fillna(0)
    
    # Add new features to the DataFrame
    df_reduced[f'{sensor}_rolling_mean_{window_size}'] = rolling_mean
    df_reduced[f'{sensor}_rolling_std_{window_size}'] = rolling_std

# Calculate the maximum cycle for each unit_id
max_cycle_per_unit = df_reduced.groupby('unit_id')['cycle'].max()
# Create the RUL column using a vectorized approach
df_reduced['RUL'] = df_reduced['unit_id'].map(max_cycle_per_unit) - df_reduced['cycle']


### model training
"""When splitting time series data, we cannot use a random split (like train_test_split with shuffle=True).
This would introduce "future" information into the training set,
as data from later cycles would be used to predict earlier cycles.
This leads to unrealistic and overly optimistic performance estimates.
Instead, we need to use a sequential split.
This means that for each unit_id, we take the earlier cycles for training and the later cycles for testing.
"""
def sequential_split(df: pandas.DataFrame, test_size=0.2):
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
      train_df = pandas.concat(train_list, axis=0)
      test_df = pandas.concat(test_list, axis=0)

      return train_df, test_df

train_df, test_df = sequential_split(df_reduced)

x_train = train_df.drop(columns=['unit_id', 'cycle', 'RUL'])
y_train = train_df['RUL']
x_test = test_df.drop(columns=['unit_id', 'cycle', 'RUL'])
y_test = test_df['RUL']


columns_to_scale = ['op_setting_1', 'op_setting_2', 'op_setting_3',
                  'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                  'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20',
                  'sensor_2_rolling_mean_10', 'sensor_2_rolling_std_10',
                  'sensor_3_rolling_mean_10', 'sensor_3_rolling_std_10', 
                  'sensor_4_rolling_mean_10', 'sensor_4_rolling_std_10', 
                  'sensor_7_rolling_mean_10', 'sensor_7_rolling_std_10', 
                  'sensor_14_rolling_mean_10', 'sensor_14_rolling_std_10', 
                  'sensor_15_rolling_mean_10', 'sensor_15_rolling_std_10', 
                  'sensor_17_rolling_mean_10', 'sensor_17_rolling_std_10', 
                  'sensor_20_rolling_mean_10', 'sensor_20_rolling_std_10']

preprocessor = ColumnTransformer(
     transformers=[
          ('scaler', StandardScaler(), columns_to_scale)
     ],
     remainder='passthrough'
)

pipeline = Pipeline([
     ('preprocessing', preprocessor),
     ("regressor", LinearRegression())
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE = {mse:.2f}')
print(f'R² score = {r2:.4f}')