import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import serial

import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

try:
    ser_com = serial.Serial(port='COM12',
                            baudrate=9600,
                            bytesize=8,
                            stopbits=1,
                            parity='N')

except:
    print("Please Check Serial Communication, Make Sure Com Cable Connected!")

warnings.filterwarnings("ignore")

df = pd.read_csv('household.csv')

print(df.head().to_string())

print("No of Lines in the given Dataset", df.shape[0])

print(df.info())

print(df.describe())

print("********************** Data Pre-Processing **********************")

def Data_Preprocessing():

    global df

    print("Null values: ", df.isnull().sum())

    plt.figure("Analysis - Looking up for the null values", figsize=(10, 5))

    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='mako')

    plt.show()

    print(df.dropna(inplace=True))

    plt.figure("Analysis - After Dropping Null Values", figsize=(10, 5))

    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

    plt.show()

    print(df[df['Humidity'] <= 30])

    plt.figure("Analysis - Temperature and Humidity before Normalization", figsize=(10, 6))

    sns.set_style('whitegrid')

    sns.lineplot(data=df, x='Humidity', y='Temperature', color='orange')

    plt.show()

    df['Humidity'] = np.where((df['Humidity'] <= 30), 30, df['Humidity'])

    print(df['Humidity'].to_string())

    plt.figure("Analysis - Temperature and Humidity After Normalization", figsize=(10, 6))

    sns.set_style('whitegrid')

    sns.lineplot(data=df, x='Humidity', y='Temperature', color='green')

    plt.show()

    Output = pd.DataFrame({'Output':[0]})

    df = pd.concat([df, Output], axis=1)

def condition_set():

    print("****************** Condition Set 1 ******************")

    df['Output'] = np.where((df['Volts'] < 1.0), int(1), df['Output'])

    print(df['Output'].to_string())

    print("****************** Condition Set 2 ******************")

    df['Output'] = np.where((df['Volts'] > 1.0), int(1), df['Output'])

    print(df['Output'].to_string())

    print("****************** Condition Set 3 ******************")

    df['Output'] = np.where((df['Temperature'] > 27.0), int(3), df['Output'])

    print(df['Output'].to_string())

    print("****************** Condition Set 4 ******************")

    df['Output'] = np.where((df['Temperature'] < 27.0) & (df['Volts'] < 1.0), int(2), df['Output'])

    print(df['Output'].to_string())

    print("****************** Condition Set 5 ******************")

    df['Output'] = np.where((df['Volts'] < 1.0) & (df['Temperature'] > 27.0), int(4), df['Output'])

    print(df['Output'].to_string())

    print(df.to_string())

Data_Preprocessing()

condition_set()

plt.figure("Analysis - Line plot Volts vs Temperature")

sns.set_style('whitegrid')

sns.lineplot(data=df, x='Volts', y='Temperature', hue='Output', palette='tab10')

plt.show()

plt.figure("Analysis - Scatter plot Volts vs Temperature")

sns.set_style('whitegrid')

sns.scatterplot(data=df, x='Volts', y='Temperature', hue='Output', palette='tab10')

plt.show()

x = df.iloc[:, 1:3]

Y = df.iloc[:, 4:]

print(x.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25,random_state=35)

print("************************* Decision Tree Classifier *************************")

tree = DecisionTreeClassifier(criterion='entropy')

tree.fit(X_train, Y_train)

pred = tree.predict(X_test)

print(pred)

accuracy = accuracy_score(pred, Y_test)

print("Accuracy Percentage: ", accuracy * 100, '\r\n')

confusion = confusion_matrix(pred, Y_test)

print("Confusion Matrix\r\n")

print(confusion)

print("************************* Random Forest Classifier *************************")

random = RandomForestClassifier(n_estimators=10, criterion='entropy')

random.fit(X_train, Y_train)

pred = random.predict(X_test)

print(pred)

accuracy = accuracy_score(pred, Y_test)

print("Accuracy: ", accuracy * 100)

confusion = confusion_matrix(pred, Y_test)

print("Confusion Matrix\r\n")

print(confusion)

print(X_test.head())

while(1):

    v = float(input("Enter Test Voltage Value:"))

    t = float(input("Enter Test Temperature Value:"))

    test_frame = pd.DataFrame({'Volts':[v], 'Temperature': [t]})

    test_predict = random.predict(test_frame)

    test_predict = np.asarray(test_predict, dtype='int')

    if(test_predict[0] == 1):
        print("Final result: Fan OFF and Light OFF")
        ser_com.write(str.encode('1'))

    if (test_predict[0] == 2):
        print("Final result: Fan OFF and Light ON")
        ser_com.write(str.encode('2'))

    if (test_predict[0] == 3):
        print("Final result: Fan ON and Light OFF")
        ser_com.write(str.encode('3'))

    if (test_predict[0] == 4):
        print("Final result: Fan ON and Light ON")
        ser_com.write(str.encode('4'))