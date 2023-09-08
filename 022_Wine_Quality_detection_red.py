import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import sleep

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('winequality_red.csv')

print(df.head().to_string())

print(df.info())

print("No of Rows in the Dataset: ", df.shape[0])

print(df.describe().to_string())

def data_analysis():

    global df

    plt.figure("Analysis - Countplot of Quality", figsize=(10, 5))

    ax = sns.countplot(data=df, x=df['quality'], palette='mako')

    ax.bar_label(ax.containers[0])

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Fixed Vs volatile Acidity", figsize=(10, 5))

    sns.lineplot(data=df, x=df['fixed acidity'], y=df['volatile acidity'], palette='icefire',
                 hue=df['quality'], marker='*', color='blue')

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Residual Sugar Vs chlorides", figsize=(10, 5))

    sns.lineplot(data=df, x=df['residual sugar'], y=df['chlorides'], palette='mako',
                 hue=df['quality'], marker='o', color='blue')

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Scatter plot for citric acid", figsize=(10, 5))

    sns.scatterplot(data=df, x=df['citric acid'], y=df['pH'], palette='viridis', hue=df['quality'])

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Co-relation in Data", figsize=(10, 6))

    sns.heatmap(data=df.corr(), yticklabels=False, cmap='viridis', annot=True)

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Looking up for the null values", figsize=(10, 6))

    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - Alcohol content vs quality", figsize=(10, 5))

    sns.lineplot(data=df, x=df['quality'], y=df['alcohol'], marker='o', color='orange')

    sns.lineplot(data=df, x=df['pH'], y=df['alcohol'], marker='*', color='blue')

    plt.tight_layout()

    plt.show()

data_analysis()

x = df.drop(['quality'], axis=1)

Y = df['quality']

print(x.head().to_string())

print(Y.head().to_string())

X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=35)

def log_reg():

    global X_train, Y_train, X_test, Y_test

    log = LogisticRegression(solver='lbfgs', max_iter=3000)

    log.fit(X_train, Y_train)

    y_pred = log.predict(X_test)

    print(y_pred)

    acc = accuracy_score(Y_test, y_pred)

    matrix = confusion_matrix(Y_test, y_pred)

    print("Accuracy Score: ", acc*100)

    print("Confusion Matrix: \r\n", matrix)


def decision_tree():

    global X_train, Y_train, X_test, Y_test

    dec = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=11, max_depth=11)

    dec.fit(X_train, Y_train)

    y_pred = dec.predict(X_test)

    print(y_pred)

    acc = accuracy_score(Y_test, y_pred)

    matrix = confusion_matrix(Y_test, y_pred)

    print("Accuracy Score: ", acc*100)

    print("Confusion Matrix: \r\n", matrix)

def ran_for():

    global X_train, Y_train, X_test, Y_test

    ran = RandomForestClassifier(n_estimators=11, criterion='entropy')

    ran.fit(X_train, Y_train)

    y_pred = ran.predict(X_test)

    print(y_pred)

    acc = accuracy_score(Y_test, y_pred)

    matrix = confusion_matrix(Y_test, y_pred)

    print("Accuracy Score: ", acc*100)

    print("Confusion Matrix: \r\n", matrix)

log_reg()

decision_tree()

ran_for()