import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pandas.read_csv('train_FD001.txt', sep='\s+', header=None,
                    names=['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                          'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                          'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
                          'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                          'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                          'sensor_21'])

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Display descriptive statistics
for i in range(21):
    print(f"sensor{i+1}:")
    print(df[f"sensor_{i+1}"].describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Correlation heatmap
# plt.figure(figsize=(12, 10))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# Plot distributions for all sensors
# for sensor in sensor_columns:
#     plt.figure(figsize=(8, 5))
#     sns.histplot(data=df[sensor], kde=True, bins=30)
#     plt.title(f'{sensor} Distribution')
#     plt.xlabel('Sensor Values')
#     plt.ylabel('Frequency')
#     plt.show()

# Create a directory to save the plots
# output_dir = "engine_sensor_plots"
# os.makedirs(output_dir, exist_ok=True)

# Select a few engines to visualize
# engines_to_plot = [1, 2, 3]  # Replace with desired engine IDs
# Iterate over each engine
# for engine_id in engines_to_plot:
#     engine_data = df[df['unit_id'] == engine_id]
    
#     # Plot all sensors for the current engine
#     plt.figure(figsize=(15, 10))
#     for sensor in [col for col in df.columns if 'sensor' in col]:
#         plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
    
#     plt.title(f'Sensor Values Over Time for Engine {engine_id}')
#     plt.xlabel('Cycle')
#     plt.ylabel('Sensor Values')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))  # Adjust legend position
#     plt.grid(True)
    
#     # Save the plot
#     plot_filename = os.path.join(output_dir, f'engine_{engine_id}_sensors.png')
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.show()

# Sensors to remove
to_remove = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Drop the columns to remove
df_filtered = df.drop(columns=to_remove)

# Calculate and print the correlation matrix
correlation_matrix = df.corr() # to change to df_filtered

# Find pairs with correlation above 0.8
threshold = 0.8
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):  # Avoid duplicate pairs
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Print the pairs with high correlation
print(f"Pairs with correlation above 0.8:{len(high_corr_pairs)}")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")


# Optional: Visualize the correlation matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# plt.title('Correlation Matrix (Without Removed Sensors)')
# plt.show()