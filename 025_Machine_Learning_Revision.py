import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("All Fine and Good to Go!")

df = pd.read_csv('Bank Customer Churn Prediction.csv')

print(df.head(5).to_string())

print(df.info())

print("No of Rows in the dataset: ", df.shape[0])

print(df.describe().to_string())

print(df.isnull().sum())

df.drop(['customer_id'], axis=1, inplace=True)

print(df.head(5).to_string())

def data_analysis():

    global df

    plt.figure("Analysis - Before Dropping Null Values", figsize=(10, 5))

    plt.title("Before Dropping Null Values")
    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
    plt.tight_layout()

    plt.show()

    df.dropna(inplace=True)

    plt.figure("Analysis - After Dropping Null Values", figsize=(10, 5))

    plt.title("After Dropping Null Values")
    sns.heatmap(data=df.isnull(), yticklabels=False, cmap='viridis')
    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - countplot for male and Female", figsize=(10, 5))

    plt.title("Number of male and Female")
    ax = sns.countplot(data=df, x=df['gender'], palette='viridis')
    plt.tight_layout()

    ax.bar_label(ax.containers[0])

    plt.show()

    plt.figure("Analysis - countplot for male and Female who churn from bank", figsize=(10, 5))

    plt.title("Number of male and Female who churn from Bank")
    ax = sns.countplot(data=df, x=df['gender'], palette='viridis', hue=df['churn'])

    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.tight_layout()

    plt.show()

    plt.figure("Analysis - countplot of Exact Tenure of the customers", figsize=(10, 5))

    plt.title("Exact Tenure")
    ax = sns.countplot(data=df, x=df['tenure'], palette='icefire')
    ax.bar_label(ax.containers[0])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers with credit card", figsize=(10, 5))

    plt.title("credit card")
    ax = sns.countplot(data=df, x=df['credit_card'], palette='icefire')
    ax.bar_label(ax.containers[0])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers with credit card who churn", figsize=(10, 5))

    plt.title("credit card customers who churn from bank")
    ax = sns.countplot(data=df, x=df['credit_card'], palette='icefire', hue='churn')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers of diffrent countries who churn", figsize=(10, 5))

    plt.title("customers of diffrent countries who churn")
    ax = sns.countplot(data=df, x=df['country'], palette='icefire', hue='churn')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers of diffrent countries who active", figsize=(10, 5))

    plt.title("customers of diffrent countries who active")
    ax = sns.countplot(data=df, x=df['country'], palette='mako', hue='active_member')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers with respect to product numbers", figsize=(10, 5))

    plt.title("customers with respect to product numbers")
    ax = sns.countplot(data=df, x=df['products_number'], palette='mako')
    ax.bar_label(ax.containers[0])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - countplot of customers with respect to product numbers who churn", figsize=(10, 5))

    plt.title("customers with respect to product numbers")
    ax = sns.countplot(data=df, x=df['products_number'], palette='mako', hue='churn')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

    plt.tight_layout()
    plt.show()

    plt.figure("Analysis - credit score vs estimated salary, balance", figsize=(10, 5))

    plt.title("Credit score with salary")

    sns.lineplot(data=df, x='credit_score', y='estimated_salary', color='orange', label='Salary')
    sns.lineplot(data=df, x='credit_score', y='balance', color='green', label='Balance')

    plt.show()

def data_shape():

    global df

    print("************* Data Shaping for prdictions *************")

    df['country'] = df['country'].map({"France": 1, "Spain": 2, "Germany": 3})
    print(df.head(5).to_string())

    df['gender'] = df['gender'].map({"Female": 1, "Male": 2})
    print(df.head(5).to_string())

def predict():

    global df

    X = df.drop(['churn'], axis=1)
    print(X.head().to_string())

    y = df['churn']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("************* Logistic Regression *************\r\n")

    log = LogisticRegression(solver='lbfgs')
    log.fit(X_train, Y_train)
    y_pred = log.predict(X_test)

    acc_score = accuracy_score(Y_test, y_pred)
    print("Accuracy Score: ", acc_score)

    con = confusion_matrix(Y_test, y_pred)
    print("Confusin Matrix: \r\n\n", con)

    print("************* Decision Tree Classifier *************\r\n")

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=9, max_leaf_nodes=6, random_state=42)
    tree.fit(X_train, Y_train)
    y_pred = tree.predict(X_test)

    acc_score = accuracy_score(Y_test, y_pred)
    print("Accuracy Score: ", acc_score)

    con = confusion_matrix(Y_test, y_pred)
    print("Confusin Matrix: \r\n\n", con)

    print("************* Random Forest Classifier *************\r\n")

    ensemble = RandomForestClassifier(criterion='entropy', n_estimators=10)
    ensemble.fit(X_train, Y_train)
    y_pred = ensemble.predict(X_test)

    acc_score = accuracy_score(Y_test, y_pred)
    print("Accuracy Score: ", acc_score)

    con = confusion_matrix(Y_test, y_pred)
    print("Confusin Matrix: \r\n\n", con)

data_analysis()
data_shape()
predict()