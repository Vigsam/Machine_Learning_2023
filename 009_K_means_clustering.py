import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from time import sleep

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("mall_customers.csv")

print(df.head().to_string())

print("Shape: \r\n", df.shape)

print("************************** Information **************************\r\n")

print(df.info())

print("\r\n")

print("No of Lines: ")

print(len(df['CustomerID'].index))

print("************************** Checking Null Values **************************\r\n")

print(df.isnull().sum())

sns.pairplot(data=df, vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], hue='Gender', palette='icefire')
plt.show()

plt.figure("Analysis - Checking Null Values", figsize=(10, 5))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='mako')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Scatter Plot for Age vs Spending Score")

sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender', palette='viridis')
plt.tight_layout()
plt.show()

df.drop(['CustomerID'], axis=1, inplace=True)
print(df.head().to_string())

X = df.iloc[:,[2, 3]]
print(X.head().to_string())

data = []

for i in range(1, 12):
    k = KMeans(n_clusters=i, init="k-means++", random_state=42)
    k.fit(X)

    data.append(k.inertia_)

print(data)

plt.figure("Analysis - Finding Elbow Point")
sns.lineplot(x=range(1, 12), y=data, marker='o', color='green')
plt.show()

k = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_k = k.fit_predict(X)

print(y_k)