import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

# Set display option to show all columns
pd.set_option('display.max_columns', None)

# Path to the Parquet file
file_path = 'yellow_tripdata_2024-09.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Select relevant numerical features
selected_features = df[['trip_distance', 'fare_amount']]

# Remove missing values
selected_features = selected_features.dropna()

# Remove extreme outliers using IQR method
Q1 = selected_features.quantile(0.25)
Q3 = selected_features.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
filtered_features = selected_features[~((selected_features < (Q1 - 1.5 * IQR)) | (selected_features > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize the data using Min-Max normalization
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(filtered_features), columns=filtered_features.columns)

# Experiment with different values of branching_factor and threshold
configurations = [
    {'branching_factor': 5, 'threshold': 0.01},
    {'branching_factor': 100, 'threshold': 0.5}
]

for config in configurations:
    # Apply the Birch clustering algorithm with the current configuration
    birch_model = Birch(n_clusters=3, branching_factor=config['branching_factor'], threshold=config['threshold'])
    clusters = birch_model.fit_predict(normalized_features)

    # Add the cluster labels to the DataFrame
    normalized_features[f'cluster_{config["branching_factor"]}_{config["threshold"]}'] = clusters

    # Visualize the clustering results
    plt.figure(figsize=(10, 6))
    plt.scatter(normalized_features['trip_distance'], normalized_features['fare_amount'], c=clusters, cmap='viridis', marker='o')
    plt.title(f'Birch Clustering (branching_factor={config["branching_factor"]}, threshold={config["threshold"]})')
    plt.xlabel('Normalized Trip Distance')
    plt.ylabel('Normalized Fare Amount')
    plt.colorbar(label='Cluster')
    plt.show()