#Packages
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

#Nltk functions
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

#Initiating Lemmatizer
lemmatizer = WordNetLemmatizer()

#Pre-compiled Regex
#Strips structural metadata out before tokenizing text
STAGE_DIRECTIONS_REGEX = re.compile(r'\[.*?\]', re.IGNORECASE)#Strips [Operator Instructions]
ANNOTATIONS_REGEX = re.compile(r'\(.*?\)', re.IGNORECASE)#Strips (laughter) or (ph-phonetic spelling)
#Below patterns also applied to the news dataset
USA_PATTERN = re.compile(r'\b(u\.s\.a\.|u\.s\.a|usa|u\.s\.|u\.s|u\s+s)\b', re.IGNORECASE)
ABV_PROTECT_RE = re.compile(r'(?<=[a-z])\.(?=[a-z])')
DIGIT_MASK_RE = re.compile(r'\b\d+(?:\.\d+)?(?:m|b|k|%)?\b')
CLEAN_ALPHA_NUM_RE = re.compile(r'[^a-z0-9\s]')
MULTIPLE_SPACES_REGEX = re.compile(r'\s+')

#Stopword and Rescue strategy
rescue_list = {
    'up', 'down', 'above', 'below', 'against', 'growth', 'fall','yoy',
    'rise', 'drop', 'gain', 'loss', 'high', 'low', 'higher', 'lower','top', 'bottom',
    'not', 'no', 'never', 'none', 'neither', 'nor', 'but','neutral'
    'off', 'under', 'over', 'against', 'only','growth', 'gain', 'loss'
}

base_stopwords = set(stopwords.words('english'))
standard_stopwords = base_stopwords - rescue_list

#Transcript spoken text junk filters
transcript_jargon = {
    'operator', 'analyst', 'executive', 'officer', 'chairman', 'ceo', 'cfo','turn','call','moderator','first','come'
    'thank', 'thanks', 'welcome', 'everyone', 'good', 'morning', 'afternoon','conference','line','answer',
    'evening', 'ready', 'question', 'conclude', 'session', 'hand', 'over','remarks','quarter','management',
    'understand', 'certainly', 'definitely', 'actually', 'basically', 'going','joining','hi','today','hello',
    'yes', 'listen', 'right', 'okay', 'guy', 'guys', 'well', 'oh', 'uh', 'um','hey','next','really','see',
    'sir', 'gentlemen', 'lady', 'ladies','standing', 'listen', 'mode', 'speaker', 'presentation', 'gentleman',
    'recorded', 'time', 'mr', 'please', 'go', 'ahead','speaker', 'presentation', 'like', 'discus','patience'
}
#Adding noise artifacts and single letters to the final ban list
#'wa' is added to catch the common WordNet lemmatization error of 'was'
noise_artifacts = {'would', 'could', 'also', 'wa', 'ha', 'get', 'take'}
alphabet_noise = set(string.ascii_lowercase) - {'a', 'i'}

FINAL_STOPWORDS = standard_stopwords.union(transcript_jargon).union(noise_artifacts).union(alphabet_noise)


# Master Pre-procesing Function
def process_transcript(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Stripping non-spoken transcript metadata
    text = STAGE_DIRECTIONS_REGEX.sub(' ', text)
    text = ANNOTATIONS_REGEX.sub(' ', text)

    # Regex Pipeline Sequence - Similar applied to news data.
    text = text.lower()
    text = USA_PATTERN.sub('usa', text)
    text = ABV_PROTECT_RE.sub('', text)
    text = re.sub(r'\byear[- ]over[- ]year\b', 'yoy', text, flags=re.IGNORECASE)
    text = re.sub(r'\by[- ]o[- ]y\b', 'yoy', text, flags=re.IGNORECASE)
    text = DIGIT_MASK_RE.sub('num', text)  # 10.5m' -> 'num'
    text = CLEAN_ALPHA_NUM_RE.sub(' ', text)
    text = MULTIPLE_SPACES_REGEX.sub(' ', text).strip()

    # Tokenization, Lemmatization, and Alphanumeric Filtering
    raw_tokens = text.split()
    clean_tokens = []

    for token in raw_tokens:
        # Catches past-tense verbs before they get mangled into noun roots
        if token in ['was', 'were']:
            lemma = 'be'
        else:
            lemma = lemmatizer.lemmatize(token)

        if lemma == 'num':
            clean_tokens.append(lemma)
            continue

        if lemma in rescue_list:
            clean_tokens.append(lemma)
            continue

        if lemma.isalnum() and len(lemma) >= 2 and lemma not in FINAL_STOPWORDS:
            clean_tokens.append(lemma)

    return " ".join(clean_tokens)

#Input Data Paths
summary = "../Data/Transcripts/ECT_Summary.csv"
granular = "../Data/Transcripts/ECT_Granular.csv"

#Loading Data
df_summary = pd.read_csv(summary)
df_granular = pd.read_csv(granular)

#Filling Missing Values
df_summary = df_summary.fillna("neutral")

#Pre-Processing Execution
df_summary[['proc_prepared_remarks','proc_qa']] = df_summary[['prepared_remarks_text','qa_text']].map(process_transcript)
df_granular['proc_text'] = df_granular['text'].map(process_transcript)

#Adding processed counts to the df summary dataset
df_summary['remarks_proc_count'] = df_summary['proc_prepared_remarks'].apply(lambda x:len(x.split()))
df_summary['qa_proc_count'] = df_summary['proc_qa'].apply(lambda x:len(x.split()))

#Adding processed counts to the df granular dataset
df_granular['proc_text_count'] = df_granular['proc_text'].apply(lambda x:len(x.split()))

#Filtering unwanted rows to form the final granular dataset
granular_final = df_granular[(df_granular['proc_text_count'] >= 10) & (df_granular['role']!= 'moderator')].copy()

#Saving Processed Data
output_path1 = '../Data/Transcripts/Summary_Processed.csv'
output_path2 = '../Data/Transcripts/Granular_Processed.csv'
df_summary.to_csv(output_path1, index=False)
granular_final.to_csv(output_path2, index=False)