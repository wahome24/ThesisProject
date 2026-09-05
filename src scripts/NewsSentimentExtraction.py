#Packages
import pandas as pd
import joblib

#Input Data Paths
path_A = "../Data/News/DatasetA_Processed.csv"
path_B = "../Data/News/DatasetB_Processed.csv"

#Loading Pre-Processed Data
dataset_A = pd.read_csv(path_A)
dataset_B = pd.read_csv(path_B)

#Creating new dataframes to store only relevant columns
#P represent processed
P_dataset_A = dataset_A[['Date','Stock_symbol','P_Article_title','P_Article']]
P_dataset_B = dataset_B[['Date','Stock_symbol','relevance_score','P_Article_title','P_Article']]

#Loading the pre-trained components from the Financial Phrasebank Analysis
tfidf = joblib.load('tfidf_vectorizer.pkl')
svm = joblib.load('svm_sentiment_scorer.pkl')
lr = joblib.load('logistic_regression_scorer.pkl')

#SVM Sentiment Classification - Dataset A
#Ensuring text columns are clean string arrays
X_titles_A = P_dataset_A['P_Article_title'].astype(str).tolist()
X_articles_A = P_dataset_A['P_Article'].astype(str).tolist()

#Vectorizing the text columns into sparse matrices
X_titles_vecA = tfidf.transform(X_titles_A)
X_articles_vecA = tfidf.transform(X_articles_A)

#Sentiment Scoring - Article Titles - SVM
P_dataset_A['title_svm_label'] = svm.predict(X_titles_vecA)
title_probs = svm.predict_proba(X_titles_vecA)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_A['title_svm_net_sentiment'] = title_probs[:, 2] - title_probs[:, 0]

#Sentiment Scoring - Articles- SVM
P_dataset_A['article_svm_label'] = svm.predict(X_articles_vecA)
article_probs = svm.predict_proba(X_articles_vecA)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_A['article_svm_net_sentiment'] = article_probs[:, 2] - article_probs[:, 0]

#SVM Sentiment Classification - Dataset B
#Ensuring text columns are clean string arrays
X_titles_B = P_dataset_B['P_Article_title'].astype(str).tolist()
X_articles_B = P_dataset_B['P_Article'].astype(str).tolist()

#Vectorizing the text columns into sparse matrices
X_titles_vecB = tfidf.transform(X_titles_B)
X_articles_vecB = tfidf.transform(X_articles_B)

#Sentiment Scoring - Article Titles - SVM
P_dataset_B['title_svm_label'] = svm.predict(X_titles_vecB)
title_probs = svm.predict_proba(X_titles_vecB)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_B['title_svm_net_sentiment'] = title_probs[:, 2] - title_probs[:, 0]

#Sentiment Scoring - Articles- SVM
P_dataset_B['article_svm_label'] = svm.predict(X_articles_vecB)
article_probs = svm.predict_proba(X_articles_vecB)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_B['article_svm_net_sentiment'] = article_probs[:, 2] - article_probs[:, 0]

#Logistic Regression Sentiment Classification - Dataset A
#Vectorization already done so only sentiment scoring will be executed

#Sentiment Scoring - Article Titles - LR
P_dataset_A['title_lr_label'] = lr.predict(X_titles_vecA)
title_lr_probs = lr.predict_proba(X_titles_vecA)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_A['title_lr_net_sentiment'] = title_lr_probs[:, 2] - title_lr_probs[:, 0]

#Sentiment Scoring - Articles - LR
P_dataset_A['article_lr_label'] = lr.predict(X_articles_vecA)
article_lr_probs = lr.predict_proba(X_articles_vecA)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_A['article_lr_net_sentiment'] = article_lr_probs[:, 2] - article_lr_probs[:, 0]

#Logistic Regression Sentiment Classification - Dataset B
#Vectorization already done so only sentiment scoring will be executed

#Sentiment Scoring - Article Titles - LR
P_dataset_B['title_lr_label'] = lr.predict(X_titles_vecB)
title_lr_probs = lr.predict_proba(X_titles_vecB)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_B['title_lr_net_sentiment'] = title_lr_probs[:, 2] - title_lr_probs[:, 0]

#Sentiment Scoring - Articles - LR
P_dataset_B['article_lr_label'] = lr.predict(X_articles_vecB)
article_lr_probs = lr.predict_proba(X_articles_vecB)
#Calculating continuous Net Probability Score (Pos - Neg)
P_dataset_B['article_lr_net_sentiment'] = article_lr_probs[:, 2] - article_lr_probs[:, 0]

#Saving Data with Sentiment Scores
output_path1 = '../Data/News/DatasetA_Sentiment_Score.csv'
output_path2 = '../Data/News/DatasetB_Sentiment_Score.csv'
P_dataset_A.to_csv(output_path1, index=False)
P_dataset_B.to_csv(output_path2, index=False)
