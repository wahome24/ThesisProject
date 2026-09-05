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
nltk.download('punkt', quiet=True)

#Input Data Paths
path_A = "../Data/News/News_Dataset A.csv"
path_B = "../Data/News/News_Dataset B.csv"

#Initiating Lemmatizer
lemmatizer = WordNetLemmatizer()


# Stopword and Rescue strategy
def get_final_stops():
    stop_words = set(stopwords.words('english'))

    # Rescue lIST : Critical for Financial Sentiment
    # Will be removed from the stopword list so they are NOT deleted.
    rescue_list = {
        'not', 'no', 'never', 'none', 'neither', 'nor', 'but',
        'up', 'down', 'above', 'below', 'higher', 'lower', 'high', 'low',
        'off', 'under', 'over', 'against', 'only',
        'growth', 'gain', 'loss',
    }

    for word in rescue_list:
        stop_words.discard(word)

    # Adding noise artifacts and single letters to the final ban list
    # 'wa' is added to catch the common WordNet lemmatization error of 'was'
    noise_artifacts = {'would', 'could', 'also', 'wa', 'ha', 'get', 'take'}
    alphabet_noise = set(string.ascii_lowercase) - {'a', 'i'}

    final_stops = stop_words.union(noise_artifacts).union(alphabet_noise)
    return final_stops


FINAL_STOPS = get_final_stops()

#Pre-compiled Regex
#USA Pattern: Normalizes all variants into 'usa'
USA_PATTERN = re.compile(r'\b(u\.s\.a\.|u\.s\.a|usa|u\.s\.|u\.s|u\s+s)\b', re.IGNORECASE)

#Abbreviation Collapser: i.e 'i.m.f.' -> 'imf'
ABV_PROTECT_RE = re.compile(r'(?<=[a-z])\.(?=[a-z])')

#Noise Cleaner: Keeps letters and numbers
CLEAN_ALPHA_NUM_RE = re.compile(r'[^a-z0-9\s]')

#Digit Masking: Turns specific numbers into a generic 'num' feature
DIGIT_MASK_RE = re.compile(r'\b\d+(?:\.\d+)?(?:m|b|k|%)?\b')


# Master Pre-processing Function
def preprocess_fin_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Case Folding
    text = text.lower()

    # USA Normalization
    text = USA_PATTERN.sub('placeholderusa', text)

    # Collapsing Abbreviations
    text = ABV_PROTECT_RE.sub('', text)

    # Metric Masking (Preserves context of magnitude without feature sparsity)
    # This turns '5.2%' or '100m' into 'num'
    text = DIGIT_MASK_RE.sub('num', text)

    # Removes Punctuation/Special Characters
    text = CLEAN_ALPHA_NUM_RE.sub(' ', text)

    # Whitespace Tokenization
    tokens = text.split()

    # Lemmatization and Final Stopword Filtering
    cleaned_tokens = []
    for w in tokens:
        w_lem = lemmatizer.lemmatize(w)

        if w_lem not in FINAL_STOPS:
            # Restore USA placeholder
            if w_lem == 'placeholderusa':
                cleaned_tokens.append('usa')
            else:
                cleaned_tokens.append(w_lem)

    return " ".join(cleaned_tokens)


#Loading Data
dataset_A = pd.read_csv(path_A)
dataset_B = pd.read_csv(path_B)

#List of columns to process
cols_to_process = ['Article_title', 'Article', 'Lsa_summary', 'Textrank_summary']
new_cols = ['P_Article_title', 'P_Article', 'P_LSA', 'P_Textrank_']

#Execution
dataset_A[new_cols] = dataset_A[cols_to_process].map(preprocess_fin_text)
dataset_B[new_cols] = dataset_B[cols_to_process].map(preprocess_fin_text)

#Saving Processed Data
output_path1 = '../Data/News/DatasetA_Processed.csv'
output_path2 = '../Data/News/DatasetB_Processed.csv'
dataset_A.to_csv(output_path1, index=False)
dataset_B.to_csv(output_path2, index=False)
