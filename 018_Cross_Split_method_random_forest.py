import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Iris.csv')

print(df.head())

print(df.info())

print(df.describe())

print("No of rows in a Dataset: ", df.shape[0])

plt.figure('Analysis - Count plot based on species', figsize=(10, 5))

sns.countplot(data=df, x='Species', palette='viridis')

plt.show()

plt.figure('Analysis - Sepal Length and Width', figsize=(10, 5))

sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')

plt.show()

plt.figure('Analysis - Petal Length and Width', figsize=(10, 5))

sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')

plt.show()

df.drop(['Id'], inplace=True, axis=1)

X = df.iloc[:, 0:4]

df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

y = df['Species']

print(X.head())

print(y.head(20))

kfold = model_selection.KFold(n_splits=10)

for train, test in kfold.split(X):
    print(train, test)

random = RandomForestClassifier(n_estimators=35, random_state=7, criterion='entropy')

print("Accuracy: ", model_selection.cross_val_score(random, X, y, scoring='accuracy', cv=10).mean()*100)