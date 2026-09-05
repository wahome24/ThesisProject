#Package Import
import pandas as pd
import re

# Slicing Function
def process_news(title, full_text, max_words=2250):
    """Combines news title and article body into a structured, unified string
    optimized for GPT-4o Mini, capping the body safely at 2250 words."""

    # Clean and normalize inputs safely
    title_str = str(title).strip() if pd.notna(title) else ""
    text_str = str(full_text).strip() if pd.notna(full_text) else ""

    # Text normalization: Strip web clutter while preserving casing/punctuation
    text_str = re.sub(r'https?://\S+|www\.\S+', '', text_str)
    text_str = re.sub(r'<.*?>', '', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)

    # Tokenize by space to measure absolute word lengths
    words = text_str.split(' ')

    # Removes empty string artifacts caused by split handling
    words = [w for w in words if w]

    # Global Window Selection: Captures up to the absolute 2250 word limit
    # If the text is short (e.g., 200 words), words[:max_words] safely returns all 200 words without error.
    context_words = words[:max_words]
    body_payload = ' '.join(context_words)

    # Building high-density structural text input template
    if title_str:
        final_input = f"TITLE: {title_str}\n\nBODY: {body_payload}"
    else:
        final_input = f"BODY: {body_payload}"

    return final_input.strip()

#PreProcessing
def preprocessing_pipeline(df, ticker_col='Stock_symbol', headline_col='Article_title', text_col='Article'):
    df['gpt_input'] = df.apply(
        lambda row: process_news(
            title=row[headline_col],
            full_text=row[text_col],
            max_words=2250
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
dataset_A['gpt_input_len'] = dataset_A['gpt_input'].apply(lambda x:len(x.split()))
dataset_B['gpt_input_len'] = dataset_B['gpt_input'].apply(lambda x:len(x.split()))

#Saving Processed Data
output_path1 = '../Data/News/GPT_DatasetA_Processed.csv'
output_path2 = '../Data/News/GPT_DatasetB_Processed.csv'
dataset_A.to_csv(output_path1, index=False)
dataset_B.to_csv(output_path2, index=False)