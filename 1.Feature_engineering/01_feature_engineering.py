import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import numpy as np


# replaces pyfinance.ols.PandasRollingOLS (no longer maintained)
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# get data

# The data is get from the folder Data which have two types of data: Price data (stock price) and Financial report data (Balance sheet, Income statement, Cash flow statement)
# Read price data in parquet format
price = pd.read_parquet('../Data/Data_price/vnstock_quote.parquet')

# Extract close price but keep date and ticker
price = price[['date', 'quote', 'close']]

# Using MultiIndex to set date and ticker as index
price = price.set_index(['quote', 'date'])

# Convert to float
price["close"] = price["close"].astype(float)
price.describe()

## Calculate alpha factors
# Calculate daily return
price['return'] = price.groupby(level='quote')['close'].pct_change()
price['return'] = price['return'].round(2)

# Drop missing values
price = price.dropna()

# Calculate 12-month rolling volatility
price['volatility_12m'] = price.groupby(level='quote')['return'].rolling(20*12).apply(lambda x: np.nanstd(x)).reset_index(level=0, drop=True)

# Calculate 12-month rolling beta
#price['beta'] = price.groupby(level='quote')['return'].rolling(12*20).apply(lambda x: sm.OLS(x[1:], sm.add_constant(x[:-1])).fit().params[1]).reset_index(level=0, drop=True)

# Calculate from 1-month to 12-month rolling return
for i in range(1, 13):
    price[f'return_{i}m'] = price.groupby(level='quote')['close'].pct_change(i*20)

# Calculate 12-month rolling Sharpe ratio
price['sharpe_12m'] = price['return_12m'] / price['volatility_12m']

# Calculate 12-month rolling momentum
for lag in [2,3,6,9,12]:
    price[f'momentum_{lag}'] = price[f'return_{lag}m'].sub(price.return_1m)
price[f'momentum_3_12'] = price['return_12m'].sub(price.return_3m)

# Date indicator
# Transform date as string to datetime
price['date'] = pd.to_datetime(price['date'], format='%Y-%m-%d')
price['year'] =  pd.to_datetime(price.index.get_level_values('date'), format='%Y-%m-%d').year
price['month'] =  pd.to_datetime(price.index.get_level_values('date'), format='%Y-%m-%d').month

# Lagged return of daily return from 1 to 7
for i in range(1, 8):
    price[f'return_lag{i}'] = price.groupby(level='quote')['return'].shift(i)

# Target holding period return from 1 to 12 months
for i in range(1, 13):
    price[f'target_return_{i}m'] = price.groupby(level='quote')['return'].shift(-i*20)

# Drop missing values
price = price.dropna()

# Save to parquet format
price.to_parquet('../Data/Data_price/vnstock_alpha.parquet')








