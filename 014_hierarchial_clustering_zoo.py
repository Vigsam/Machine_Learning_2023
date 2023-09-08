import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import sleep

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('zoo.csv')

print(df.head().to_string())

print(df.shape)

print(df.info())

print(df.describe().to_string())

print(df['class_type'].unique())

print(df['class_type'].value_counts())

print(df.isnull().sum())

plt.figure("Analysis - Checking up for the null values", figsize=(10, 5))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='mako')

plt.tight_layout()

plt.show()

plt.figure("Analysis - Looking up for the Value counts", figsize=(10, 5))

sns.countplot(data=df, x='class_type', palette='viridis')

plt.show()

features = df.values[:, 1:17]

print(features.shape)

model = AgglomerativeClustering(n_clusters=7, linkage='average', affinity='cosine')

model.fit(features)

print(model.labels_)

labels = df['class_type']

print(labels)

labels = labels - 1

score = mean_squared_error(labels, model.labels_)

print(score)

abs_error = np.sqrt(score)

print(abs_error)