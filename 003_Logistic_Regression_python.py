from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv("suv_data.csv")

X = df.iloc[:,[2, 3]].values
y = df.iloc[:,[4]].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

reg = LogisticRegression(random_state=0)
reg.fit(X_train, y_train.ravel())

prediction = reg.predict(X_test)

print(prediction)

score = accuracy_score(y_test, prediction)

print(score)

confusion = confusion_matrix(y_test, prediction)
print(confusion)