#Package Import
import pandas as pd
import re

#File Paths
input_path =  r"C:\Users\USER\Downloads\nasdaq_exteral_data.csv"
output_path = "../Data/News/news_data.csv"

#Define Tickers and Aliases
#This ensures all possible ticker formats are picked in the regex search.
target_stocks = {
    "INTC": ["INTC", "INTC.O", "INTC.OQ", "q:INTC"],
    "WMT": ["WMT", "WMT.N", "WMT.K", "q:WMT"],
    "GOOGL": ["GOOGL", "GOOG", "GOOGL.O", "GOOG.O"],
    "GS": ["GS", "GS.N", "GS.K", "q:GS"],
    "DAL": ["DAL", "DAL.N", "DAL:US","DE"],
    "CMCSA": ["CMCSA", "CMCSA.O", "CMCSA.OQ", "q:CMCSA"],
    "BX": ["BX", "BX.N", "BX.K", "q:BX"],
    "ADSK": ["ADSK", "ADSK.O"],
    "HPE": ["HPE", "HPE.N"],
    "PYPL": ["PYPL", "PYPL.O"],
    "APTV": ["APTV", "APTV.N"],
    "FOX": ["FOX", "FOXA", "FOXB", "FOX.O", "FOXA.O"],
    "XOM": ["XOM", "XOM.N"],
    "BKR": ["BKR", "BKR.O"],
    "APA": ["APA", "APA.O"]
}

#Regex
all_aliases = [alias for sublist in target_stocks.values() for alias in sublist]
#Escaping periods for regex optimization
escaped_aliases = [re.escape(a) for a in all_aliases]
regex_pattern = r'^(' + '|'.join(escaped_aliases) + r')$'

#Chunking
chunk_size = 150000
first_chunk = True

#Data Filter
for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
    # Standardizing Symbol Column
    chunk['Stock_symbol'] = chunk['Stock_symbol'].astype(str).str.strip().str.upper()

    # Filtering using Regex exact match against the alias list
    mask = chunk['Stock_symbol'].str.match(regex_pattern, na=False)
    filtered = chunk[mask].copy()

    if not filtered.empty:
        #Normalizing Map aliases back to the master Ticker (e.g., JPM.N -> JPM)
        for master_ticker, aliases in target_stocks.items():
            filtered.loc[filtered['Stock_symbol'].isin(aliases), 'Stock_symbol'] = master_ticker

        #Saving the filtered data
        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        filtered.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
