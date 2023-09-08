import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('headbrain.csv')

x = df['Head Size(cm^3)'].values.reshape(-1, 1)
y = df['Brain Weight(grams)'].values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(x, y)

y_prediction = reg.predict(x)

plt.title("Linear Regression Model")
plt.xlabel("Head Size in cm^3")
plt.ylabel("Brain Weight in grams")

plt.scatter(x, y, color='orange', label="Head Size vs Brain Weight")
plt.plot(x, y_prediction, color='green', label="Head Size vs Regression Line for Brain Weight", marker='o')

plt.legend()

plt.show()