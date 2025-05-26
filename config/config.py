# filepath: config/config.py
WINDOW_SIZE = 10
RUL_CAP = 125

# Define all sensor columns and operational settings
ALL_SENSORS = [f'sensor_{i}' for i in range(1, 22)]
OP_SETTINGS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# Define sensors to remove
LOW_VARIANCE_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
HIGHLY_CORRELATED_SENSORS = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_9', 'sensor_21']

# Dataset-specific configurations
DATASETS = {
    "FD001": {
        "TRAIN_PATH": "data/train_FD001.txt",
        "TEST_PATH": "data/test_FD001.txt",
        "RUL_PATH": "data/RUL_FD001.txt",
        "LOW_VARIANCE_SENSORS": LOW_VARIANCE_SENSORS,
        "HIGHLY_CORRELATED_SENSORS": HIGHLY_CORRELATED_SENSORS
    },
    "FD002": {
        "TRAIN_PATH": "data/train_FD002.txt",
        "TEST_PATH": "data/test_FD002.txt",
        "RUL_PATH": "data/RUL_FD002.txt",
        "LOW_VARIANCE_SENSORS": LOW_VARIANCE_SENSORS,
        "HIGHLY_CORRELATED_SENSORS": HIGHLY_CORRELATED_SENSORS
    },
    "FD003": {
        "TRAIN_PATH": "data/train_FD003.txt",
        "TEST_PATH": "data/test_FD003.txt",
        "RUL_PATH": "data/RUL_FD003.txt",
        "LOW_VARIANCE_SENSORS": LOW_VARIANCE_SENSORS,
        "HIGHLY_CORRELATED_SENSORS": HIGHLY_CORRELATED_SENSORS
    },
    "FD004": {
        "TRAIN_PATH": "data/train_FD004.txt",
        "TEST_PATH": "data/test_FD004.txt",
        "RUL_PATH": "data/RUL_FD004.txt",
        "LOW_VARIANCE_SENSORS": LOW_VARIANCE_SENSORS,
        "HIGHLY_CORRELATED_SENSORS": HIGHLY_CORRELATED_SENSORS
    }
}

def get_config(dataset_name):
    """Retrieve configuration for the specified dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} is not defined in the configuration.")

    dataset_config = DATASETS[dataset_name]
    to_remove = sorted(set(dataset_config["LOW_VARIANCE_SENSORS"] + dataset_config["HIGHLY_CORRELATED_SENSORS"]))
    columns_to_scale = OP_SETTINGS + [sensor for sensor in ALL_SENSORS if sensor not in to_remove]

    return {
        "TRAIN_PATH": dataset_config["TRAIN_PATH"],
        "TEST_PATH": dataset_config["TEST_PATH"],
        "RUL_PATH": dataset_config["RUL_PATH"],
        "TO_REMOVE": to_remove,
        "COLUMNS_TO_SCALE": columns_to_scale
    }