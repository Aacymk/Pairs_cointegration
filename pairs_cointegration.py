import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import statsmodels.api as sm

# Add tickers to analyze
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'AMD', 'CRM', 'INTC', 'ADBE', 'CSCO', 'QCOM',
    'ORCL', 'IBM', 'TXN', 'MU', 'NOW', 'SHOP', 'UBER']

# Must sum to 1
PCT_TRAIN = 0.75
PCT_TEST = 0.25

# Must be in YYYY-MM-DD format
START_DATE = '2020-07-01'
END_DATE = '2025-07-01'


pairs_train = []
pairs_test = []
train_data = []
test_data = []
for ticker1 in tickers:
    for ticker2 in tickers:
        try:
            data1 = yf.download(ticker1, start=START_DATE, end=END_DATE)
            data2 = yf.download(ticker2, start=START_DATE, end=END_DATE)
            if len(data1) != len(data2):
              raise ValueError
            length_train, length_test = int(PCT_TRAIN * len(data1)), int(PCT_TEST * len(data1))
            while True:
              if length_train + length_test > len(data1):
                length_test = length_test - 1
              elif length_train + length_test < len(data1):
                length_test = length_test + 1
              else:
                break
            data1, data2 = (data1['Close'][ticker1], data2['Close'][ticker2])
            train_A, test_A = (data1.head(length_train), data1.tail(length_test))
            train_B, test_B = (data2.head(length_train), data2.tail(length_test))
            X = sm.add_constant(train_B)
            model = sm.OLS(train_A ,X).fit()
            a = model.params['const']
            b = model.params[ticker2]
            res = train_A - (a + b*train_B)
            adf_stat, p_value, _, _, crit_vals, _ = adfuller(res)
            train_data.append({'ticker1': ticker1, 'ticker2': ticker2, 'ADF Stat': round(adf_stat, 4), 'P-value': round(p_value, 4)})

            if p_value < 0.05:
              pairs_train.append((ticker1, ticker2))
              X = sm.add_constant(test_B)
              model = sm.OLS(test_A ,X).fit()
              a = model.params['const']
              b = model.params[ticker2]
              res = test_A - (a + b*test_B)
              adf_stat, p_value, _, _, crit_vals, _ = adfuller(res)
              test_data.append({'ticker1': ticker1, 'ticker2': ticker2, 'ADF Stat': round(adf_stat, 4), 'P-value': round(p_value, 4)})
              if p_value < 0.05:
                pairs_test.append((ticker1, ticker2))
            else:
              X = sm.add_constant(test_B)
              model = sm.OLS(test_A ,X).fit()
              a = model.params['const']
              b = model.params[ticker2]
              res = test_A - (a + b*test_B)
              adf_stat, p_value, _, _, crit_vals, _ = adfuller(res)
              test_data.append({'ticker1': ticker1, 'ticker2': ticker2, 'ADF Stat': round(adf_stat, 4), 'P-value': round(p_value, 4)})

        except:
            print(f'Error downloading {ticker1}, {ticker2}')

print(f"Pairs in training phase: {pairs_train}\n")
print(f"Pairs in testing phase: {pairs_test}")

df1 = pd.DataFrame(train_data)
heatmap_data1 = df1.pivot(index="ticker1", columns="ticker2", values="P-value")
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data1, annot=True, cmap="coolwarm", cbar_kws={'label': 'P-value'})
plt.title("ADF Test P-values by Pair in Training Phase")
plt.show()

df2 = pd.DataFrame(test_data)
heatmap_data2 = df2.pivot(index="ticker1", columns="ticker2", values="P-value")
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data2, annot=True, cmap="coolwarm", cbar_kws={'label': 'P-value'})
plt.title("ADF Test P-values by Pair in Testing Phase")
plt.show()


