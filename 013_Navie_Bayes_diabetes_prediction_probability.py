import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('diabetes.csv')
print(df.head().to_string())

print(df.info())
print(df.describe().to_string())
print("Shape of the dataset")

print(df.shape)

plt.figure("Analysis - look up for the Null Value", figsize=(10, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Pregnancies vs BMI", figsize=(10, 5))
sns.lineplot(data=df, x='Pregnancies', y='BMI')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Blood Pressure vs BMI", figsize=(10, 5))
sns.lineplot(data=df, x='Insulin', y='BMI', color='orange')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Insulin vs BMI", figsize=(10, 5))
sns.lineplot(data=df, x='Insulin', y='BMI')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Age Count plot", figsize=(14, 5))
sns.countplot(data=df, x='Age')
plt.tight_layout()
plt.show()

plt.figure("Analysis - Outcome countplot", figsize=(10, 5))
sns.countplot(data=df, x='Outcome', palette='viridis')
plt.tight_layout()
plt.show()

#X = df.drop(['Outcome'], axis=1)

X = df.drop(['Outcome', 'Pregnancies'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.fit_transform(X_test)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("\r\n***************** Accuracy Score *****************\r\n")

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\r\n***************** Confusion Matrix *****************\r\n")

confusion = confusion_matrix(y_test, y_pred)
print(confusion)

print("\r\n***************** Classification Report *****************\r\n")

classification = classification_report(y_test, y_pred)
print(classification)