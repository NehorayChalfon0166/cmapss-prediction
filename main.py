import pandas

df = pandas.read_csv('train_FD001.txt', sep='\s+', header=None,
                    names=['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                          'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                          'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
                          'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                          'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                          'sensor_21'])


# from data exloration i saw that sensors 1, 5, 10, 16, 18, 19
# are basiccly constants so have little to no effect
to_remove = ['sensor_1', 'sensor_5', 'sensor_10',
            'sensor_16','sensor_18','sensor_19']

df.drop(to_remove)
