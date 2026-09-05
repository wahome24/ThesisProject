#Packages
import pandas as pd
import re


# Cleaning
def clean_and_normalize_speaker_text(text):
    """Purifies dialogue rows by removing markdown noise and excessive whitespace."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    # Strips URL structures and HTML artifacts from transcript exports
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
    cleaned = re.sub(r'<.*?>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


# Input Structuring
def process_speaker_row(ticker, speaker, role, speaker_text, max_words=380):
    clean_ticker = str(ticker).strip().upper()
    clean_speaker = str(speaker).strip() if pd.notna(speaker) else "Unknown Speaker"
    clean_role = str(role).strip() if pd.notna(role) else "Participant"

    # Constructs an informative title header natively from row metadata
    constructed_title = f"{clean_ticker} Call - {clean_speaker} ({clean_role})"

    raw_text = clean_and_normalize_speaker_text(speaker_text)
    if not raw_text:
        return ""

    words = raw_text.split(' ')

    # Conditional Check: If row is short, pass it through natively
    if len(words) <= max_words:
        return f"TITLE: {constructed_title} | SPEAKER_CONTEXT: {raw_text}"

    # Fallback Path: Handles long rows by slicing them
    sliced_words = words[:max_words]
    sliced_text = " ".join(sliced_words)
    return f"TITLE: {constructed_title} | SLICED_SPEAKER_CONTEXT: {sliced_text}"


# Processing
def transcript_pipeline(df, ticker_col='ticker', speaker_col='speaker', role_col='role', text_col='text'):
    df['finbert_input'] = df.apply(
        lambda row: process_speaker_row(
            ticker=row[ticker_col],
            speaker=row[speaker_col],
            role=row[role_col],
            speaker_text=row[text_col],
            max_words=380  # Safe boundary limit protecting against the 512 token wall
        ), axis=1
    )
    # Drops empty rows
    initial_len = len(df)
    df = df[df['finbert_input'] != ""].reset_index(drop=True)

#Input Data Paths
granular = "../Data/Transcripts/ECT_Granular.csv"

#Loading Data
df_granular = pd.read_csv(granular)

#Execution
transcript_pipeline(df_granular)

#Dropping Moderator rows
granular_final = df_granular[df_granular['role'] != 'moderator']
granular_final = granular_final.reset_index()

#Adding Processed word counts
granular_final['finbert_input_len'] = granular_final['finbert_input'].apply(lambda x:len(x.split()))

#Saving Processed Data
output_path1 = '../Data/Transcripts/FinBERT_Granular_Processed.csv'
granular_final.to_csv(output_path1, index=False)