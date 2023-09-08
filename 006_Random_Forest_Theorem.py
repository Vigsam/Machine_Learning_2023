import numpy as np
import pandas as pd
import seaborn as sns
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

'''******************* Read csv File *******************'''

df = pd.read_csv('penguins.csv')
print(df.head(5).to_string(), '\r\n')

'''******************* Shape of the csv File *******************'''

print("Shape: ", df.shape, '\r\n')
print("Information\r\n")

'''******************* Information about the csv File *******************'''

print(df.info())

'''******************* Finding Null Values *******************'''

print("\r\n")
print(df.isnull().sum())
plt.figure("Analysis - Finding Null Values")
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

'''******************* Dropping Null Values *******************'''

df.dropna(inplace=True)

'''******************* After Dropping Null Values *******************'''

plt.figure("Analysis - After Null Values Removed")
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

print(df.info())

'''******************* Getting Dummies and dropping First *******************'''

sex = pd.get_dummies(df['sex'], drop_first=True)
print(sex.head(5))

island = pd.get_dummies(df['island'], drop_first=True)
print(island.head(5))

'''******************* Getting Dummies and dropping First *******************'''

df = pd.concat([df, sex, island], axis = 1)
print(df.head(5).to_string())

df.drop(['sex', 'island'], axis=1, inplace=True)
print(df.head(5).to_string())

'''******************* Mapping the species value *******************'''

species = df.species
print(species.head(5))

print(species.unique())

species = species.map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})
print(species.head(5))

X = df.drop(['species'], axis=1)
Y = species

print(X.head(5).to_string())
print(Y.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

classifier = RandomForestClassifier(criterion='entropy', n_estimators=7, random_state=0)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
print(prediction)

score = accuracy_score(y_test, prediction)
print(score*100)

confusion = confusion_matrix(y_test, prediction)
print(confusion)