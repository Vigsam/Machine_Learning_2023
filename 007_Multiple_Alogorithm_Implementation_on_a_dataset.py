import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from time import sleep

import warnings

warnings.filterwarnings('ignore')

''' Reading the csv File '''

df = pd.read_csv('penguins.csv')
print(df.to_string())

''' shape of the dataframe '''

print("Shape: ", df.shape, '\r\n')

''' Information of the dataframe '''

print(df.info())

''' No of Data availbale in the dataset '''

print("No of Data in a dataset: ", df['species'].index, '\r\n')

''' Data Analysis '''

df_split = df.groupby('species')

for id, frame in df_split:

    ''' Adelie '''

    if id == "Adelie":
        df_adelie = frame
        print(df_adelie.head(5).to_string(),'\n')

    ''' Chinstrap '''

    if id == "Chinstrap":
        df_chinstrap = frame
        print(df_chinstrap.head(5).to_string(),'\n')

    ''' Gentoo '''

    if id == "Gentoo":
        df_gentoo = frame
        print(df_gentoo.head(5).to_string(),'\n')

print("Data Split Happened")

plt.figure("Analysis - Species Adelie (Bill length vs Bill depth)", figsize=(12, 5))

sns.lineplot(data=df_adelie, x=df['bill_length_mm'], y=df['bill_depth_mm'], hue='island')
plt.show()

plt.figure("Analysis - Species Adelie (flipper length vs Body Mass)", figsize=(12, 5))

sns.lineplot(data=df_adelie, x=df['flipper_length_mm'], y=df['body_mass_g'], hue='island')
plt.show()

plt.figure("Analysis - Species vs body mass in g", figsize=(12, 5))

sns.barplot(data=df, x='species', y='body_mass_g', hue='island', palette='mako')
plt.show()

plt.figure("Analysis - Species vs sex", figsize=(12, 5))

sns.countplot(data=df, x='species', hue='island', palette='icefire')
plt.show()

plt.figure("Analysis - Check for null Values", figsize=(12, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

df.dropna(inplace=True)

plt.figure("Analysis - After Dropping null values", figsize=(12, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

island = pd.get_dummies(df['island'], drop_first=True)
print(island.head(2).to_string())

sex = pd.get_dummies(df['sex'], drop_first=True)
print(sex.head(2).to_string())

species = df['species'].map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})

df.drop(['island', 'sex', 'species'], axis=1, inplace=True)
print(df.head(2).to_string())

df = pd.concat([df, island, species, sex], axis=1)
print(df.head(2).to_string())

X = df.drop(['species'], axis=1)
Y = species

print(X.head(2).to_string())
print(Y.head(2).to_string())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

''' Logistic Regression '''

Regression = LogisticRegression(solver='lbfgs', max_iter=1000)

Regression.fit(X_train, y_train)
predict = Regression.predict(X_test)

print("********************* Logistic Regression *********************")

print(predict)

score = accuracy_score(y_test, predict)
confusion = confusion_matrix(y_test, predict)
classification = classification_report(y_test, predict)

print("Score: \r\n", score)
print("Confusion: \r\n", confusion)
print("Classification: \r\n", classification)

''' Decision Tree Classifier '''

tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_leaf_nodes=4, random_state=100)

tree.fit(X_train, y_train)
predict = tree.predict(X_test)

print("********************* Decision Tree Classifier *********************")

print(predict)

score = accuracy_score(y_test, predict)
confusion = confusion_matrix(y_test, predict)
classification = classification_report(y_test, predict)

print("Score: \r\n", score)
print("Confusion: \r\n", confusion)
print("Classification: \r\n", classification)

''' Random Forest Classifier '''

random = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=100)

random.fit(X_train, y_train)
predict = random.predict(X_test)

print("********************* Random Forest Classifier *********************")

print(predict)

score = accuracy_score(y_test, predict)
confusion = confusion_matrix(y_test, predict)
classification = classification_report(y_test, predict)

print("Score: \r\n", score)
print("Confusion: \r\n", confusion)
print("Classification: \r\n", classification)

''' Support Vector Theorem '''

print("********************* Random Forest Classifier *********************")

svm = SVC(random_state=100)
svm.fit(X_train, y_train)

predict = svm.predict(X_test)

print(predict)

score = accuracy_score(y_test, predict)
matrix = confusion_matrix(y_test, predict)
classification = classification_report(y_test, predict)

print("Accruacy Score: ", score*100)
print("Confusion Matrix: \r\n\n", matrix)
print("Classification Report: \r\n\n", classification)