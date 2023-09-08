from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score, mean_squared_error

df = pd.read_csv('advertising.csv')
print("\r\n" + df.head(5).to_string())

df.drop(["Unnamed: 0"], inplace=True, axis=1)
print("\r\n" + df.head(2).to_string())

print("--------- Shape ---------\r\n")

print(df.shape, "\r\n")

print(df.info())

print(df.describe())

print("--------- Looking up for the Null Values ---------")

print(df.isnull().sum())

print("--------- Looking up for the Cor-relation ---------")

plt.figure("Analysis - Co-relation")

sns.heatmap(data=df.corr(), annot=True, cmap='viridis')

plt.show()

print("--------- TV vs Sales ---------")

plt.figure("Analysis - TV vs Sales")

plt.xlabel("TV")

plt.ylabel("Sales")

sns.lineplot(data=df, x=df["TV"], y=df["Sales"])

plt.show()

print("--------- Radio vs Sales ---------")

plt.figure("Analysis - Radio vs Sales")

plt.xlabel("TV")

plt.ylabel("Sales")

sns.lineplot(data=df, x=df["Radio"], y=df["Sales"])

plt.show()

print("--------- Newspaper vs Sales ---------")

plt.figure("Analysis - Newspaper vs Sales")

plt.xlabel("Newspaper")

plt.ylabel("Sales")

sns.lineplot(data=df, x=df["Radio"], y=df["Sales"])

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)

axes[0].scatter(df["TV"], df["Sales"], color='green')
axes[1].scatter(df["Radio"], df["Sales"], color='blue')
axes[2].scatter(df["Newspaper"], df["Sales"], color='orange')

axes[0].set_ylabel("Sales")
axes[0].set_xlabel("TV")
axes[1].set_xlabel("Radio")
axes[2].set_xlabel("Newspaper")

plt.tight_layout()

plt.show()

X = df.drop(["Sales", "Newspaper", "Radio"], axis=1)
y = df["Sales"]

print(X.head(2).to_string())
print(y.head(2).to_string())

reg = LinearRegression()
reg.fit(X, y)

print(reg.intercept_)
print(reg.coef_)

X_new = pd.DataFrame({"TV": [50, 100, 200]})

pred = reg.predict(X_new)
print(pred)