#Packages
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

#Nltk functions
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

#Loading Dataset
data_path = "../Data/PhraseBank/Sentences_AllAgree.txt"
phrasebank_df = pd.read_csv(data_path,sep='@',names=['sentence', 'sentiment'],encoding='latin1',
    engine='python')

#Function to add numerical labels
def labels(x):
    if x.lower() == 'positive':
        return 2
    elif x.lower() == 'neutral':
        return 1
    else:
        return 0

phrasebank_df['sent_label'] = phrasebank_df['sentiment'].apply(labels)

#Pre-Processing
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


#Master Pre-procesing Function
def preprocess_fin_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Case Folding
    text = text.lower()

    # USA Normalization
    text = USA_PATTERN.sub('placeholderusa', text)

    # Collapsing identical concepts - i.e mln > million
    text = re.sub(r'\b(mn|mln)\b', 'million', text)
    text = re.sub(r'\beur\b', 'euro', text)

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

#Running on the Phrasebank dataset
phrasebank_df['P_sentence'] = phrasebank_df['sentence'].apply(preprocess_fin_text)

#Train Test Split
# Features and Targets
X = phrasebank_df['P_sentence'].astype(str)
y = phrasebank_df['sent_label']

#Stratified split for class imbalance control
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#Vectorization via TF-IDF
#Initializing the vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

#Fitting the data
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

#Modeling
#Support Vector Machine (Linear Kernel works best for sparse text vectors)
svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train_vec, y_train)

#Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)

#Predictions
svm.predict(X_test_vec)
lr.predict(X_test_vec)

#Saving vectorizer and Models
#TF-IDF Vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

#Sentiment scorers
joblib.dump(svm, 'svm_sentiment_scorer.pkl')
joblib.dump(lr, 'logistic_regression_scorer.pkl')
