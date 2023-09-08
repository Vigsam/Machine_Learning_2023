from time import sleep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("titanic.csv")

print(df.head().to_string())

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

plt.figure("Analysis - Looking up for the Null Values")

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

plt.show()

df.drop(['Cabin', 'PassengerId'], inplace=True, axis=1)

print(df.head().to_string())

df.dropna(inplace=True)

plt.figure("Analysis - Looking up for the Null Values")

sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

plt.show()

plt.figure("Analysis - Sex counter plot")

sns.countplot(data=df, x=df['Sex'], hue='Survived', palette='mako')

plt.show()

plt.figure("Analysis - Embarked counter plot")

sns.countplot(data=df, x=df['Embarked'], hue='Survived', palette='viridis')

plt.show()

plt.figure("Analysis - Survived counter plot")

sns.countplot(data=df, x=df['Survived'])

plt.show()

plt.figure("Analysis - Passenger class counter plot")

sns.countplot(data=df, x=df['Pclass'], hue='Survived', palette='icefire')

plt.show()

print(" ************ Data Understanding using Visualization ************ ")

print("Number of Male and Female: \r\n", df['Sex'].value_counts())

print("Survived: \r\n", df['Survived'].value_counts())

print("Passenger Class: \r\n", df['Pclass'].value_counts())

print("Parents and Children: \r\n", df['Parch'].value_counts())

print("Embarked: \r\n", df['Embarked'].value_counts())

Embarked_c = pd.get_dummies(df['Embarked'], drop_first=True)
print(Embarked_c.head())

Pclass_c = pd.get_dummies(df['Pclass'], drop_first=True)
print(Pclass_c.head())

Sex_c = pd.get_dummies(df['Sex'], drop_first=True)
print(Sex_c.head())

df.drop(['Pclass', 'SibSp', 'Embarked', 'Name', 'Ticket', 'Fare', 'Sex'], inplace=True, axis=1)

df = pd.concat([df, Embarked_c, Pclass_c, Sex_c], axis=1)

print(df['male'])

print(df.head().to_string())

plt.figure("Analysis - Co-relation")

sns.heatmap(data=df.corr(), annot=True, cmap='viridis')

plt.show()

X = df.drop(['Survived'], axis=1)

y = df['Survived']

print(X.head())

print(y.head())

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("X_test: ", X_test.shape)

print("Y_test: ", y_test.shape)

reg = LogisticRegression(solver='lbfgs', max_iter=1000)

reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

print(y_predict)

accuracy = accuracy_score(y_test, y_predict)

print("Accuracy: ", accuracy)

print("Confusion Matrix: \r\n\n", confusion_matrix(y_test, y_predict))

print("Classification report: \r\n\n", classification_report(y_test, y_predict))