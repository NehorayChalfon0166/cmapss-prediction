# filepath: config/config.py
WINDOW_SIZE = 10
RUL_CAP = 125

TRAIN_PATH = 'data/train_FD001.txt'
TEST_PATH = 'data/test_FD001.txt'
RUL_PATH = 'data/RUL_FD001.txt'

LOW_VARIANCE_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21']
TO_REMOVE = sorted(set(LOW_VARIANCE_SENSORS + HIGHLY_CORRELATED_SENSORS))

COLUMNS_TO_SCALE = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
]
COLUMNS_TO_SCALE = [col for col in COLUMNS_TO_SCALE if col not in TO_REMOVE]