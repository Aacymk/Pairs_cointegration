# Pairs Trading Stationarity Test

This project analyzes a selected list of tickers to identify cointegration.
It uses yfinance data to create a regression between tickers over a train and a test period.
It is not a backtest of any kind, but more of a framework made to be expanded upon

## Overview 
User sets a list of stocks, a train/test split, and a start/end date.  
The program itterates through every combination of stocks within the train period, checking for cointegration by testing stationarity on residuals from a regression of stock2 onto stock1.
If the training p-value < 0.05, the pair will be considered cointegrated over the train period and is then evaluated on the testing period using the same metrics.
If the pair is still cointegrated it is added to a new list which is outputted as a result.
There is also a heatmap of p-value data for both train and test periods.  


## Requirements
Install dependencies with:
pip install -r requirements.txt

## Usage 
Clone the repository:
git clone https://github.com/Aacymk/Pairs_cointegration.git
cd Pairs_cointegration

Run the script: python pairs_cointegration.py
