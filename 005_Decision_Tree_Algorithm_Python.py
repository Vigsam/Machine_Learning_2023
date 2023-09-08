import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import sleep
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import serial

''' ********************************* Decision Tree ********************************* '''
try:
    serial_communication_1 = serial.Serial(port='COM11',
                                         baudrate=9600,
                                         bytesize=8,
                                         parity= 'N',
                                         stopbits=1
                                         )

except:
    print("Serial Communication 1 Not OK")

try:
    serial_communication_2 = serial.Serial(port='COM3',
                                           baudrate=9600,
                                           bytesize=8,
                                           parity='N',
                                           stopbits=1
                                           )
except:
    print("Serial Communication 2 Not OK")

''' ********************************* Decision Tree ********************************* 

Decision Tree is the tree shaped Algorithm which used to determine the course of action. Each branch of tree provides 
the possible decision, occurances and reaction.

Entrophy: It is the measure of randomness of unpredictability in the dataset.
Information Gain: Decrease in entrophy after a data split is known as the Information Gain.

'''

df = pd.read_csv('loan_data.csv')
print(df.head(2).to_string())

plt.figure("Analysis - Loan Purpose", figsize=(12, 5))
sns.countplot(data=df, x='purpose', palette='mako')

plt.show()

plt.figure("Analysis - Loan Purpose and Not Fully Paid", figsize=(12, 5))
sns.countplot(data=df, x='purpose', hue='not.fully.paid', palette='icefire')

plt.show()

sns.set_style('darkgrid')
plt.figure("Analysis - Fico mapping with credit policy", figsize=(12, 5))

plt.hist(df['fico'].loc[df['credit.policy'] == 1], bins=30, label='credit policy=1')
plt.hist(df['fico'].loc[df['credit.policy'] == 0], bins=30, label='credit policy=0')

plt.legend()
plt.show()

sns.set_style('whitegrid')
plt.figure("Analysis - Fico mapping with Not fully paid", figsize=(12, 5))

plt.hist(df['fico'].loc[df['not.fully.paid'] == 1], bins=30, label='not fully paid = 1')
plt.hist(df['fico'].loc[df['not.fully.paid'] == 0], bins=30, alpha=0.5, label='fully paid = 0')

plt.legend()
plt.show()

'''

plt.figure('Analysis - Fico and int', figsize=(10, 6))

sns.jointplot(data=df, x='fico', y='int.rate')
plt.xlabel("FICO")
plt.ylabel("Interest Rate")
plt.show()

'''

purpose_c = pd.get_dummies(df['purpose'], drop_first=True)
df = pd.concat([df, purpose_c], axis=1)

df.drop('purpose', axis=1, inplace=True)

print(df.head(2).to_string())

print(df.isnull().sum())

plt.figure("Analysis - Heat Map for Finding Null Values")

sns.heatmap(df.isnull(), yticklabels=False, cmap='mako')
plt.show()

y = df['not.fully.paid']
X = df.drop('not.fully.paid', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)

reg.fit(X_train, y_train)

prediction = reg.predict(X_test)
print(prediction)

score = accuracy_score(y_test, prediction) * 50
print("Accuracy Score: ", score)

matrix = confusion_matrix(y_test, prediction)
print("Matrix: ", matrix)

def read():
    r = serial_communication.readline()
    print(r)

s1 = "Score: 4\r\n"
s2 = "Score: 3\r\n"
s3 = "Score: 2\r\n"
s4 = "Score: 1\r\n"

if(score > 90):
    try:
        serial_communication_1.write(b'4')
        serial_communication_1.write(s1.encode())
        print("Serial Write: 4")

    except:
        print("Serial 1 Error")

    try:
        serial_communication_2.write(b'4')
        serial_communication_2.write(s1.encode())
        print("Serial Write: 4")

    except:
        print("Serial 2 Error")


if((score < 90) & (score > 75)):
    try:
        serial_communication_1.write(b'3')
        serial_communication_1.write(s2.encode())
        print("Serial Write: 3")

    except:
        print("Serial 1 Error")

    try:
        serial_communication_2.write(b'3')
        serial_communication_2.write(s2.encode())
        print("Serial Write: 3")

    except:
        print("Serial 2 Error")


if ((score < 75) & (score > 60)):
    try:
        serial_communication_1.write(b'2')
        serial_communication_1.write(s3.encode())
        print("Serial Write: 2")
    except:
        print("Serial 1 Error")

    try:
        serial_communication_2.write(b'2')
        serial_communication_2.write(s2.encode())
        print("Serial Write: 2")

    except:
        print("Serial 1 Error")


if(score < 60):
    try:
        serial_communication_1.write(b'1')
        serial_communication_1.write(s4.encode())
        print("Serial Write: 1")
    except:
        print("Serial 1 Error")

    try:
        serial_communication_2.write(b'1')
        serial_communication_2.write(s4.encode())
        print("Serial Write: 1")
    except:
        print("Serial 2 Error")