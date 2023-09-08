import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

'''   Logistic Regression

Logistic Regression is the Statistical Model which helps to find the relationship between 
Dependant and Independant Variables in which both are in form of discrete.

'''


df = pd.read_csv('titanic.csv')

print("\r\n*********************** 10 Data Head ***********************\r\n")

print(df.head(10).to_string())

no_of_passengers = len(df['PassengerId'].index)

print("\r\nTotal No of Passengers: ", no_of_passengers)

print("\r\n********************* Analysis Part *********************\r\n")

plt.figure("Analysis - Survived!")
sns.countplot(data=df, x="Survived")

plt.show()

plt.figure("Analysis - Survived Seggregation based on Sex")
sns.countplot(data=df, x="Survived", hue="Sex")

plt.show()

plt.figure("Analysis - Age Wise")

df['Age'].plot.hist()
plt.show()

plt.figure("Analysis - Fare Wise")

df['Fare'].plot.hist()
plt.show()

plt.figure("Analysis - Siblings Boarded The Ship")
sns.countplot(data=df, x="SibSp")
plt.show()

plt.figure("Analysis - Parents and Children Who Boarded The Ship")
sns.countplot(data=df, x="Parch")
plt.show()

''' Data Wrangling '''

print("\r\n*********************** Data Wrangling! ***********************\r\n")

print(df.isnull().to_string())    #True -> Null

print(df.isnull().sum())    #Counts of Null Values in each and every column

plt.figure("Analysis - Missing Data Heat Map")
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

plt.figure("Analysis - Passenger Class vs Age")
sns.boxplot(x="Pclass", y="Age", data=df)
plt.show()

print(df.head(5).to_string())

''' Droping an Axis - Cabin due to insufficient data in the Column '''

df.drop("Cabin", axis=1, inplace=True)
print(df.head(5).to_string())

df.dropna(inplace=True)
print(df.to_string())

print("\r\nAfter Data Wrangling - In Counts: ", len(df['PassengerId'].index))

plt.figure("Analysis - After Data Wrangling\r\n")
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

print(df.isnull().sum())

sex = pd.get_dummies(df['Sex'])

sex = pd.get_dummies(df['Sex'], drop_first=True)
print(sex.head(5))

embarked = pd.get_dummies(df['Embarked'], drop_first=True)
print(embarked.head(5))

pcl = pd.get_dummies(df['Pclass'], drop_first=True)
print(pcl.head(5))

df = pd.concat([df, sex, embarked, pcl], axis=1)
print(df.head(2).to_string())

df.drop(['Pclass', 'Sex', 'Ticket', 'Embarked', 'Name'], axis=1, inplace=True)
print(df.head(2).to_string())

X = df.drop('Survived', axis=1)
y = df['Survived']

X.columns = X.columns.astype(str)

print(X.head(2))
print(y.head(2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logmodel = LogisticRegression(solver='lbfgs', max_iter=800)

print(logmodel.fit(X_train, y_train))

prediction = logmodel.predict(X_test)
print(prediction)

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))

print(accuracy_score(y_test, prediction))