import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

df = pd.read_csv('breast_cancer_data.csv')

print(df.head().to_string())

print("\r\n****************** Shape of the Dataset ******************\r\n")

print(df.shape)
print(df.info())

print("\r\n****************** Description of the Dataset ******************\r\n")

print(df.describe().to_string())

print("\r\n****************** Looking up for the Null Values ******************\r\n")

print(df.isnull().sum())

plt.figure("Analysis - Looking up for the Null Values", figsize=(10, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.tight_layout()
plt.show()

diag = df['diagnosis']

print(diag.to_string())

df_1 = diag.map({'M':1.0, 'B':0.0})
print(df_1.to_string())

print("\r\n****************** Dropping the Null Values ******************\r\n")

df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1, inplace=True)

plt.figure("Analysis - Looking up for the Null Values", figsize=(10, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.tight_layout()
plt.show()

print(df.isnull().sum())

print("\r\n****************** Data Pre-processing ******************\r\n")

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

print(scaled_data.shape)

pca = PCA(n_components=3)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(x_pca.shape)

print(x_pca)

plt.figure("Analysis - Principal Components")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.scatter(x_pca[:,0], x_pca[:,1], c=df_1 ,cmap='viridis')
plt.show()