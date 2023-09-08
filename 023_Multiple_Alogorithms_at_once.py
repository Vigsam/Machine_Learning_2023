import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('penguins.csv')

print(df.head().to_string())

print(df.info())

print(df.describe().to_string())

print("Number of Rows in the Dataset: ", df.shape[0], '\r\n')

print("Full Data Set\r\n\n")

print(df.to_string())

def data_pre_process():

    global df

    print("************* Data Pre-process *************")

    print("Null Values: ", df.isnull().sum())

    plt.figure("Analysis - Heatmap before dropping Null Values", figsize=(10, 5))

    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')

    plt.tight_layout()

    plt.show()

    df.dropna(inplace=True)

    plt.figure("Analysis - Heatmap before dropping Null Values", figsize=(10, 5))

    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='icefire')

    plt.tight_layout()

    plt.show()

def data_analysis():

    global df

    print("************* Data Analysis *************")

    plt.figure("Analysis - Count plot for Species", figsize=(10, 5))

    ax = sns.countplot(data=df, x='species', palette='viridis', hue='island')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    ax.bar_label(ax.containers[2])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - Count plot for island", figsize=(10, 5))

    ax = sns.countplot(data=df, x='island', palette='mako', hue='species')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    ax.bar_label(ax.containers[2])

    plt.tight_layout()
    plt.show()

data_pre_process()

data_analysis()

df['species'] = df['species'].map({"Adelie": 1, "Chinstrap": 2, "Gentoo": 3})

df['island'] = df['island'].map({"Torgersen": 1, "Biscoe": 2, "Dream": 3})

df['sex'] = df['sex'].map({"MALE": 1, "FEMALE": 2})

print(df.to_string())

X = df.drop(['species'], axis=1)

y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train: ", X_train.shape[0])

print("y_train: ",y_train.shape[0])

print("X_test: ",X_test.shape[0])

print("y_test: ",y_test.shape[0])

def log_reg():

    print("Logistic Regression")

    global X_train, X_test, y_train, y_test

    log = LogisticRegression(solver='lbfgs', max_iter=266)

    log.fit(X_train, y_train)

    y_pred = log.predict(X_test)

    acc = accuracy_score(y_pred, y_test)

    print("Accuarcy : ", acc*100)

    con = confusion_matrix(y_pred, y_test)

    print("Confusion Matrix: \r\n", con)

def dec_tree():

    global X_train, X_test, y_train, y_test

    print("Decision Tree")

    tree = DecisionTreeClassifier(criterion='entropy')

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    acc = accuracy_score(y_pred, y_test)

    print("Accuarcy : ", acc * 100)

    con = confusion_matrix(y_pred, y_test)

    print("Confusion Matrix: \r\n", con)

def ran_for():

    global X_train, X_test, y_train, y_test

    print("Random Forest")

    forest = RandomForestClassifier(criterion='entropy', random_state=42)

    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_test)

    acc = accuracy_score(y_pred, y_test)

    print("Accuarcy : ", acc * 100)

    con = confusion_matrix(y_pred, y_test)

    print("Confusion Matrix: \r\n", con)

log_reg()

dec_tree()

ran_for()