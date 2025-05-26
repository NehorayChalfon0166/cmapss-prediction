# filepath: config/config.py
WINDOW_SIZE = 10
RUL_CAP = 125
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21']

DATASETS = {
    "FD001": {
        "TRAIN_PATH": "data/train_FD001.txt",
        "TEST_PATH": "data/test_FD001.txt",
        "RUL_PATH": "data/RUL_FD001.txt",
        "TO_REMOVE": ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19', 'sensor_11',
                    'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21'],
        "COLUMNS_TO_SCALE": [
            'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
            'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
        ]
    },
    "FD002": {
        "TRAIN_PATH": "data/train_FD002.txt",
        "TEST_PATH": "data/test_FD002.txt",
        "RUL_PATH": "data/RUL_FD002.txt",
        "TO_REMOVE": ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19', 'sensor_11',
                    'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21'],
        "COLUMNS_TO_SCALE": [
            'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
            'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
        ]
    },
    "FD003": {
        "TRAIN_PATH": "data/train_FD003.txt",
        "TEST_PATH": "data/test_FD003.txt",
        "RUL_PATH": "data/RUL_FD003.txt",
        "TO_REMOVE": ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19', 'sensor_11',
                    'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21'],
        "COLUMNS_TO_SCALE": [
            'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
            'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
        ]
    },
    "FD004": {
        "TRAIN_PATH": "data/train_FD004.txt",
        "TEST_PATH": "data/test_FD004.txt",
        "RUL_PATH": "data/RUL_FD004.txt",
        "TO_REMOVE": ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19', 'sensor_11',
                    'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21'],
        "COLUMNS_TO_SCALE": [
            'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
            'sensor_8', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20'
        ]
    }
}

def get_config(dataset_name):
    """Retrieve configuration for the specified dataset."""
    return DATASETS.get(dataset_name, {})