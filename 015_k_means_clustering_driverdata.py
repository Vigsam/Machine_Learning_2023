from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

df = pd.read_csv('driver-data.csv')

print(df.head().to_string())

print(df.info())

print(df.describe().to_string())

print("Shape of the dataset:", df.shape)

df.drop(['id'], axis=1, inplace=True)

print(df.head().to_string())

print("Analysis - Looking up for the null values")

print(df.isnull().sum())

plt.figure("Analysis - Looking up for the null values", figsize=(12, 5))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

plt.show()

cluster = KMeans(n_clusters=2)

cluster.fit(df)

print(cluster.labels_)

print(cluster.cluster_centers_)

df['cluster'] = cluster.labels_

plt.figure('Analysis - for cluster=2', figsize=(12, 5))

sns.scatterplot(data=df, x='mean_dist_day', y='mean_over_speed_perc', hue='cluster', palette='mako')

plt.show()

df.drop(['cluster'], inplace=True, axis=1)

print(df.head().to_string())

cluster = KMeans(n_clusters=4)

cluster.fit(df)

print(cluster.labels_)

print(cluster.cluster_centers_)

df['cluster'] = cluster.labels_

plt.figure("Analysis - no of clusters=4", figsize=(12, 5))

sns.scatterplot(data=df, x='mean_dist_day', y='mean_over_speed_perc', hue='cluster', palette='viridis')

plt.show()