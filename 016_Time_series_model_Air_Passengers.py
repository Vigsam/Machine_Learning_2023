import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('airpassengers.csv')
print(df.head())
print(df.info())

print(df.describe())
print("Number of lines in the dataset: ", df.shape[0])
df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)

print(df.head(5))
df.set_index(['Month'], inplace=True)
print(df.head())

plt.figure("Analysis - Month vs passengers")

plt.plot(df, color='green', label='original')

plt.legend()
plt.tight_layout()

plt.show()

df_mean = df.rolling(window=12).mean()
df_std = df.rolling(window=12).std()

df_mean.dropna(inplace=True)
df_std.dropna(inplace=True)

plt.figure("Analysis - Original, mean and Avg")

plt.plot(df, color='green', label='original')
plt.plot(df_mean, color='red', label='mean')
plt.plot(df_std, color='blue', label='std')

plt.legend()
plt.tight_layout()
plt.show()

df_test = adfuller(df['#Passengers'], autolag='AIC')

df_out = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'No of Observations Used'])

for key,value in df_test[4].items():
    df_out['Critical Value (%s)' %key] = value

print(df_out)

df_log = np.log(df)

moving_avg = df_log.rolling(window=12).mean()
moving_std = df_log.rolling(window=12).std()

moving_avg.dropna(inplace=True)
moving_std.dropna(inplace=True)

plt.figure('Analysis - Log applied, mean moving, std moving')

plt.plot(df_log, color='green', label='log')
plt.plot(moving_avg, color='red', label='moving_avg')
plt.plot(moving_std, color='orange', label='moving_std')

plt.legend()
plt.tight_layout()
plt.show()

df_log_minus_moving_avg = df_log - moving_avg
df_log_minus_moving_avg.dropna(inplace=True)
print(df_log_minus_moving_avg.head(12))

def test_stationary(timeseries):

    moving_avg = timeseries.rolling(window=12).mean()
    moving_std = timeseries.rolling(window=12).std()

    moving_avg.dropna(inplace=True)
    moving_std.dropna(inplace=True)

    plt.figure('Analysis - df_log_minus_avg, mean moving, std moving')

    plt.plot(timeseries, color='green', label='df_log_minus_moving_avg')
    plt.plot(moving_avg, color='red', label='moving_avg')
    plt.plot(moving_std, color='orange', label='moving_std')

    plt.legend()
    plt.tight_layout()
    plt.show()

test_stationary(df_log_minus_moving_avg)