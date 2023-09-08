import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import numpy as np

import time

df = pd.read_excel('gas_generator.xlsx')

df.dropna(inplace=True)

print(df.to_string(), '\r\n')

print(df.info(), '\r\n')

print(df.describe().to_string(),'\r\n')

print("No of Lines in the given dataset", df.shape[0], '\r\n')
print(df.head().to_string(), '\r\n')


print("Maximum Initial Recorded: ", df['INITIAL'].max(), '\r\n')
print("Minimum Initial Recorded: ", df['INITIAL'].min(), '\r\n')

print("Maximum Final Recorded: ",df['FINAL'].max(), '\r\n')
print("Maximum Final Recorded: ",df['FINAL'].min(), '\r\n')

print("Maximum H2O2 Power Consumption for the Day: ", df['TOTAL'].max(), '\r\n')
print("Minimum H2O2 Power Consumption for the Day: ", df['TOTAL'].min(), '\r\n')

print("Average H2O2 Power Consumption: ", df['TOTAL'].mean(), '\r\n')

print("Median: ", df['TOTAL'].median(), '\r\n')

print("Analysis - Line Graph")

plt.figure("Analysis - Date vs Total", figsize=(10, 6))

sns.lineplot(data=df, x=df['DATE'], y=df['TOTAL'], color='orange', hue=df['MONTH'], palette='hls')

plt.title('Analysis - April to July H2O2 Power Consumption Line Plot in kWh')

plt.show()

plt.figure("Analysis - Bar Plot", figsize=(12, 6))

sns.barplot(data=df, x=df['DATE'], y=df['TOTAL'], hue=df['MONTH'], palette='hls')

plt.title('Analysis - April to July H2O2 Power Consumption Bar Plot in kWh')

plt.show()

print(df.iloc[0:31, 4:5])

df_april = df.iloc[0:30, 4:5]
print("********************* April *********************")
print(df_april.to_string())

df_may = df.iloc[30:61, 4:5]
print("********************* May *********************")
print(df_may.to_string())

df_june = df.iloc[61:91, 4:5]
print("********************* June *********************")
print(df_june.to_string())

df_july = df.iloc[91:116, 4:5]
print("********************* July *********************")
print(df_july.to_string())


df_april.reset_index(inplace=True)
df_may.reset_index(inplace=True)
df_june.reset_index(inplace=True)
df_july.reset_index(inplace=True)

print("\r\n*************************** APRIL MONTH H2O2 POWER CONSUMPTION ***************************\r\n")

print("Total Consumption: ", df_april['TOTAL'].sum(), 'kWh','\r\n')
print("Maximum for the Day: ", df_april['TOTAL'].max(), 'kWh','\r\n')
print("Minimum for the Day: ", df_april['TOTAL'].min(), 'kWh','\r\n')
print("Average Consumption: ", df_april['TOTAL'].mean(), 'kWh','\r\n')

print("\r\n*************************** MAY MONTH H2O2 POWER CONSUMPTION ***************************\r\n")

print("Total Consumption: ", df_may['TOTAL'].sum(), 'kWh','\r\n')
print("Maximum for the Day: ", df_may['TOTAL'].max(), 'kWh','\r\n')
print("Minimum for the Day: ", df_may['TOTAL'].min(), 'kWh','\r\n')
print("Average Consumption: ", df_may['TOTAL'].mean(), 'kWh','\r\n')

print("\r\n*************************** JUNE MONTH H2O2 POWER CONSUMPTION ***************************\r\n")

print("Total Consumption: ", df_june['TOTAL'].sum(), 'kWh','\r\n')
print("Maximum for the Day: ", df_june['TOTAL'].max(), 'kWh','\r\n')
print("Minimum for the Day: ", df_june['TOTAL'].min(), 'kWh','\r\n')
print("Average Consumption: ", df_june['TOTAL'].mean(), 'kWh','\r\n')

print("\r\n*************************** JULY MONTH H2O2 POWER CONSUMPTION ***************************\r\n")

print("Total Consumption: ", df_july['TOTAL'].sum(), 'kWh','\r\n')
print("Maximum for the Day: ", df_july['TOTAL'].max(), 'kWh','\r\n')
print("Minimum for the Day: ", df_july['TOTAL'].min(), 'kWh','\r\n')
print("Average Consumption: ", df_july['TOTAL'].mean(), 'kWh','\r\n')

df_cum = pd.DataFrame({"APRIL": df_april['TOTAL'],"MAY": df_may['TOTAL'],"JUNE": df_june['TOTAL'], "JULY": df_july['TOTAL']})

print(df_cum.to_string())

plt.figure("Analysis - Barplot in June, July 2023", figsize=(10, 6))

ax = sns.scatterplot(data=df_cum, palette='flare')

plt.title('Analysis - Scatterplot H2O2 Power Consumption for April to July 2023 in kWh')

plt.show()