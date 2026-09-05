#Importing required packages
import yfinance as yf
import pandas as pd
import numpy as np

#Setup
tickers = ['ADSK', 'APA', 'APTV', 'BKR', 'BX', 'CMCSA', 'DAL', 'FOX', 'GOOGL','GS', 'HPE', 'INTC', 'PYPL', 'WMT', 'XOM']
benchmark = ['^GSPC']  #S&P 500 Index
all_tickers = tickers + benchmark

#Downloading the data
raw_data = yf.download(all_tickers, start="2019-01-01", end="2024-12-31",auto_adjust=False)['Adj Close']

#Calculating log returns: ln(Price_t / Price_t-1)
log_returns = np.log(raw_data / raw_data.shift(1))

#Structuring the final price dataset
processed_list = []

for ticker in tickers:
    #DataFrame for each stock
    ticker_df = pd.DataFrame({
        'Date': log_returns.index,
        'Stock_symbol': ticker,
        'Adj_Close': raw_data[ticker],
        'Log_Return': log_returns[ticker],
        'Market_Return': log_returns['^GSPC']
    })

    #Calculating Market-Adjusted Return (Excess Return)
    #Excess Return = Stock Log Return - S&P 500 Log Return
    ticker_df['Excess_Return'] = ticker_df['Log_Return'] - ticker_df['Market_Return']

    #Target Variable
    #Since News/Transcripts at Time T predict price at Time T+1, we shift by -1
    ticker_df['Target_Next_Day_Return'] = ticker_df['Excess_Return'].shift(-1)

    processed_list.append(ticker_df)

#Consolidating and saving the price data
price_df = pd.concat(processed_list).dropna()
output_path = '../Data/Price/Project_PriceData.csv'
price_df.to_csv(output_path, index=False)