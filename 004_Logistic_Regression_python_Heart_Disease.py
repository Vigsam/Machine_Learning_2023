import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('framingham.csv')
print(df.head(2).to_string())

print(df.isnull().sum())

plt.figure("Analysis - Before Data Cleaning", figsize=(12, 5))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

df.dropna(inplace=True)

plt.figure("Analysis - After Data Cleaning", figsize=(12, 5))

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='rocket')
plt.show()

plt.figure("Analysis - Age wise in numbers!", figsize=(12, 5))

sns.countplot(data=df, x='age', palette='viridis')
plt.show()

plt.figure("Analysis - Education Wise - Grade", figsize=(12, 5))

sns.countplot(data=df, x='education', palette='mako')
plt.show()

plt.figure("Analysis - Education wise currentSmokers", figsize=(12, 5))

sns.countplot(data=df, x='education', hue='currentSmoker', palette='icefire')
plt.show()

plt.figure("Analysis - Education wise currentSmokers/cigs per day", figsize=(12, 5))

sns.barplot(data=df, x='age', y='cigsPerDay', palette='viridis')
plt.show()

plt.figure("Analysis - age and cholestrol Values", figsize=(12, 5))

sns.lineplot(data=df, x='age', y='totChol', hue='currentSmoker')
plt.show()

plt.figure("Analysis - age and BMI correlation with currentSmoker", figsize=(12, 5))

sns.lineplot(data=df, x='age', y='BMI', hue='currentSmoker')
plt.show()

plt.figure("Analysis - age and glucose with diabetes", figsize=(12, 5))

sns.lineplot(data=df, x='age', y='glucose', hue='diabetes')
plt.show()

plt.figure("Analysis - age and heartRate with respect to diabetes", figsize=(12, 5))

sns.lineplot(data=df, x='age', y='heartRate', hue='diabetes')
plt.show()

X = df.drop(['education'], axis = 1)

X.columns = X.columns.astype(str)

y = df.iloc[:, 15]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.10, random_state=1)

reg = LogisticRegression(solver='lbfgs', max_iter=1500)

reg.fit(X_train, y_train)

prediction = reg.predict(X_test)

print("Predicted Values against Y_test: ", y_test)

score = accuracy_score(y_test, prediction)
print("Accuracy Score: ", score*100)

matrix = confusion_matrix(y_test, prediction)
print("Confusion Matrix: ", matrix)