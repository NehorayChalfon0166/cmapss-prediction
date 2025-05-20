import pandas

df = pandas.read_csv('train_FD001.txt', sep='\s+', header=None,
                    names=['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                          'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                          'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
                          'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                          'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                          'sensor_21'])


# from data exloration i saw that sensors 1, 5, 6, 10, 16, 18, 19
# are basiccly constants so have little to no effect
# and 'sensor_11','sensor_12','sensor_8','sensor_13','sensor_9' have strong correlations and can be removed

low_variance = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
            'sensor_16','sensor_18','sensor_19']

strong_correlation = ['sensor_11','sensor_12','sensor_8','sensor_13','sensor_9', 'sensor_21']

to_remove = low_variance + strong_correlation
df_reduced = df.drop(to_remove, axis=1)

print(df_reduced.head())





# feature engineering
# Sensors to calculate rolling statistics for
# selected_sensors = ['sensor_2', 'sensor_7', 'sensor_7']
# window_size = 10
# Group by unit_id and calculate rolling statistics
# rolling_features = []
# for sensor in selected_sensors:
#     rolling_mean = df.groupby('unit_id')[sensor].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
#     rolling_std = df.groupby('unit_id')[sensor].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
#     # Replace NaN values in rolling_std with 0
#     rolling_std = rolling_std.fillna(0)
    
#     # Add new features to the DataFrame
#     df[f'{sensor}_rolling_mean_{window_size}'] = rolling_mean
#     df[f'{sensor}_rolling_std_{window_size}'] = rolling_std
