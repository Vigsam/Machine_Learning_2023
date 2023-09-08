import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')
print(df.head().head().to_string())

print("No of rows in the Dataset: ", df.shape[0])

print(df.info())

print(df.describe().to_string())

plt.figure("Analysis - Looking up for the null values", figsize=(10, 6))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='mako')
plt.tight_layout()
plt.show()

plt.figure("Analyis - Line Plot between Preganacies and Glucose", figsize=(10, 6))

sns.lineplot(data=df, x='Pregnancies', y='Glucose', hue='Outcome', palette='viridis')

plt.show()

plt.figure("Analysis - Line plot for Glucose and Blood Pressure", figsize=(10, 6))

sns.lineplot(data=df, x='Glucose', y='BloodPressure', hue='Outcome', palette='viridis')

plt.show()

plt.figure("Analysis - Count plot Pregnancies with and without diabetes", figsize=(10, 6))

sns.countplot(data=df, x='Pregnancies', palette='icefire', hue='Outcome')

plt.show()

X = df.iloc[:, 0:8]

Y = df.iloc[:, 8:]

print(X.head().to_string())
print(Y.head().to_string())

kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
adaboost = AdaBoostClassifier(n_estimators=35)

results = model_selection.cross_val_score(adaboost, X, Y, cv=kfold)

print("Accuracy of AdaBoost: ", model_selection.cross_val_score(adaboost, X, Y, cv=kfold).mean()*100)

kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
xg = XGBClassifier(n_estimators=35)

results = model_selection.cross_val_score(xg, X, Y, cv=kfold)

print("Accuracy of XGBoost: ", model_selection.cross_val_score(xg, X, Y, cv=kfold).mean()*100)