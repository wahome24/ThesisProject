#Package Import
import pandas as pd
import re
import os

TICKER_MAP = {
    "INTC":  "Intel",
    "WMT":   "Walmart",
    "GOOGL": "Google|Alphabet",
    "GS":    "Goldman Sachs",
    "DAL":   "Delta Air Lines",
    "CMCSA": "Comcast",
    "BX":    "Blackstone",
    "ADSK":  "Autodesk",
    "HPE":   "Hewlett Packard Enterprise",
    "PYPL":  "PayPal",
    "APTV":  "Aptiv",
    "FOX":   "Fox",
    "XOM":   "ExxonMobil",
    "BKR":   "Baker Hughes",
    "APA":   "Apache|APA Corporation"
}


# Text Slicing Function
def process_news(title, full_text, ticker, window_words=175):
    title_str = str(title).strip() if pd.notna(title) else ""
    text_str = str(full_text).strip() if pd.notna(full_text) else ""

    # Text normalization:Strips web noise while preserving casing and punctuation
    text_str = re.sub(r'https?://\S+|www\.\S+', '', text_str)
    text_str = re.sub(r'<.*?>', '', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)
    words = text_str.split(' ')

    # Dynamically builds the ticker search
    clean_ticker = str(ticker).strip().upper()
    search_terms = [re.escape(clean_ticker)]
    if clean_ticker in TICKER_MAP:
        search_terms.append(re.escape(TICKER_MAP[clean_ticker]))

    # Compiles a highly precise word-boundary search regex (e.g., \b(ADSK|Autodesk)\b)
    pattern = re.compile(rf"\b({'|'.join(search_terms)})\b", re.IGNORECASE)

    target_idx = -1
    for idx, word in enumerate(words):
        if pattern.search(word):
            target_idx = idx
            break  # Break at first contextual mention

    # Slicing execution block
    if target_idx == -1:
        # Fallbacks to article head if ticker/company name are missing from body text
        # Grabw exactly the first 350 words as a safe macro fallback
        context_words = words[:(window_words * 2)]
        final_input = f"TITLE: {title_str} | MACRO_CONTEXT: {' '.join(context_words)}"
    else:
        # Target localized context window found - 175 words before, 175 words after
        start_bound = max(0, target_idx - window_words)
        end_bound = min(len(words), target_idx + window_words)
        context_words = words[start_bound:end_bound]
        final_input = f"TITLE: {title_str} | TARGET_CONTEXT: {' '.join(context_words)}"

    return final_input.strip()

#PreProcessing
def preprocessing_pipeline(df, ticker_col='Stock_symbol', headline_col='Article_title', text_col='Article'):
    df['finbert_input'] = df.apply(
        lambda row: process_news(
            title=row[headline_col],
            full_text=row[text_col],
            ticker=row[ticker_col]
        ), axis=1
    )

#Input Data Paths
path_A = "../Data/News/News_Dataset A.csv"
path_B = "../Data/News/News_Dataset B.csv"

#Datasets - Cleaned News Data
dataset_A = pd.read_csv(path_A)
dataset_B = pd.read_csv(path_B)

#Dataset A Execution
preprocessing_pipeline(dataset_A)

#Dataset B Execution
preprocessing_pipeline(dataset_B)

#Adding Processed word counts
dataset_A['finbert_input_len'] = dataset_A['finbert_input'].apply(lambda x:len(x.split()))
dataset_B['finbert_input_len'] = dataset_B['finbert_input'].apply(lambda x:len(x.split()))

#Saving Processed Data
output_path1 = '../Data/News/FinBERT_DatasetA_Processed.csv'
output_path2 = '../Data/News/FinBERT_DatasetB_Processed.csv'
dataset_A.to_csv(output_path1, index=False)
dataset_B.to_csv(output_path2, index=False)

