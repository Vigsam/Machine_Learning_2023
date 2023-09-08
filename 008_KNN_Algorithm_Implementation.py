from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('diabetes.csv')
print(df.head().to_string())

print("No of Lines: ", len(df['Glucose'].index))

print("Shape")

print(df.shape)

print("Info")

print(df.info())

print("Check for Null Values - Non Visualization")

print(df.isnull().sum())

print("Check for Null Values - Visualization Technique")

plt.figure("Analysis - Check for Null-Values Visualization Technique", figsize=(10, 5))
sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
plt.show()

plt.figure('Analysis - BMI vs BloodPressure')

sns.lineplot(data=df, x="BMI", y='BloodPressure')
plt.show()

plt.figure('Analysis - BMI vs Glucose')

sns.lineplot(data=df, x="BMI", y='Glucose')
plt.show()

plt.figure('Analysis - Pregnancies vs BloodPressure')

sns.barplot(data=df, x="Pregnancies", y='BloodPressure', palette='viridis')
plt.show()

plt.figure('Analysis - Pregnancies vs BMI')

sns.barplot(data=df, x="Pregnancies", y='BMI', palette='viridis')
plt.show()

zero_detection = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'BMI']

for column in zero_detection:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

print(df.head(2).to_string())

X = df.iloc[:,0:8]
print(X.head(2).to_string())
Y = df.iloc[:,8]
print(Y.head(2).to_string())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

knn.fit(X_train, y_train)
predict = knn.predict(X_test)

print("Results:\r\n\n", predict)

score = accuracy_score(y_test, predict)
matrix = confusion_matrix(y_test, predict)
report = classification_report(y_test, predict)

print('\r\nAccuracy Score: ', score*100)
print('Confusion Matrix: \r\n', matrix)
print('Classification Report: \r\n', report)