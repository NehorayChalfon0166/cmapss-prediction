from config.config import get_config, COLUMN_NAMES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_dataset(file_path, column_names):
    """Load the dataset from a file."""
    return pd.read_csv(file_path, sep='\s+', header=None, names=column_names)

def summarize_dataset(df):
    """Print dataset summary statistics and missing values."""
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

def plot_correlation_matrix(df, output_dir=None):
    """Plot and save the correlation matrix heatmap."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()

def find_highly_correlated_features(df, threshold=0.8):
    """Identify pairs of features with high correlation."""
    correlation_matrix = df.corr()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
    return high_corr_pairs

def plot_sensor_distributions(df, sensor_columns, output_dir=None):
    """Plot and save distributions for all sensors."""
    for sensor in sensor_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df[sensor], kde=True, bins=30)
        plt.title(f'{sensor} Distribution')
        plt.xlabel('Sensor Values')
        plt.ylabel('Frequency')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{sensor}_distribution.png'))
        plt.show()

def explore_dataset(dataset_name):
    """Explore the dataset by summarizing, plotting correlations, and distributions."""
    config = get_config(dataset_name)
    if not config:
        print(f"Dataset {dataset_name} not found in configuration.")
        return

    output_dir = f"exploration_outputs/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exploring dataset: {dataset_name}")
    df = load_dataset(config["TRAIN_PATH"], COLUMN_NAMES)

    # Summarize the dataset
    summarize_dataset(df)

    # Plot correlation matrix
    plot_correlation_matrix(df, output_dir)

    # Find and print highly correlated features
    correlation_threshold = 0.8
    high_corr_pairs = find_highly_correlated_features(df, threshold=correlation_threshold)
    print(f"\nPairs with correlation above {correlation_threshold}: {len(high_corr_pairs)}")
    for pair in high_corr_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

    # Plot sensor distributions
    sensor_columns = [col for col in df.columns if 'sensor' in col]
    plot_sensor_distributions(df, sensor_columns, output_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Error: not enough arguments!")
        print("Usage: python data_exploration.py <dataset_name1> <dataset_name2> ...")
        print("Example: python data_exploration.py FD001 FD002")
    else:
        dataset_names = sys.argv[1:]
        for dataset_name in dataset_names:
            print(f"\n=== Processing {dataset_name} ===")
            explore_dataset(dataset_name)