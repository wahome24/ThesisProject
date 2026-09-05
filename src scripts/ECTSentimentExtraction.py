#Packages
import pandas as pd
import joblib

#Input Data Paths
path_A = "../Data/Transcripts/Summary_Processed.csv"
path_B = "../Data/Transcripts/Granular_Processed.csv"

#Loading Pre-Processed Data
df_summary = pd.read_csv(path_A)
df_granular = pd.read_csv(path_B)

#Creating new dataframes to store only relevant columns
#P represent processed
P_summary = df_summary[['ticker','fiscal_year', 'fiscal_quarter','calendar_date','proc_prepared_remarks','proc_qa']]
P_granular = df_granular[['ticker','fiscal_year', 'fiscal_quarter','calendar_date','section_type','role','proc_text']]

#Converting Role to a Numerical Feature - Granular dataset
#Executive = 1, Analyst = 2
role_map = {'executive': 1, 'analyst': 2}
P_granular['role_id'] = P_granular['role'].map(role_map)

#Loading the pre-trained components from the Financial Phrasebank Analysis
tfidf = joblib.load('tfidf_vectorizer.pkl')
svm = joblib.load('svm_sentiment_scorer.pkl')
lr = joblib.load('logistic_regression_scorer.pkl')

#Vectorization - Summary Dataset
#Ensuring text columns are clean string arrays
X_remarks = P_summary['proc_prepared_remarks'].astype(str).tolist()
X_qa = P_summary['proc_qa'].astype(str).tolist()
#Vectorizing the text columns into sparse matrices
X_remarks_vec = tfidf.transform(X_remarks)
X_qa_vec = tfidf.transform(X_qa)

#SVM Sentiment Classification - Summary
#Sentiment Scoring - Prepared Remarks - SVM
P_summary['remarks_svm_label'] = svm.predict(X_remarks_vec)
remarks_probs = svm.predict_proba(X_remarks_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_summary['remarks_svm_net_sentiment'] = remarks_probs[:, 2] - remarks_probs[:, 0]
#Sentiment Scoring - Q&A- SVM
P_summary['qa_svm_label'] = svm.predict(X_qa_vec)
qa_probs = svm.predict_proba(X_qa_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_summary['qa_svm_net_sentiment'] = qa_probs[:, 2] - qa_probs[:, 0]

#Vectorization - Granular Dataset
#Ensuring text columns are clean string arrays
X_text = P_granular['proc_text'].astype(str).tolist()
#Vectorizing the text column into sparse matrices
X_text_vec = tfidf.transform(X_text)

#SVM Sentiment Classification - Granular DF
#Sentiment Scoring - Processed Text - SVM
P_granular['txt_svm_label'] = svm.predict(X_text_vec)
text_probs = svm.predict_proba(X_text_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_granular['txt_svm_net_sentiment'] = text_probs[:, 2] - text_probs[:, 0]

#Logistic Regression Sentiment Classification - Summary Dataset
#Vectorization already done so only sentiment scoring will be executed

#Sentiment Scoring - Remarks - LR
P_summary['remarks_lr_label'] = lr.predict(X_remarks_vec)
remarks_lr_probs = lr.predict_proba(X_remarks_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_summary['remarks_lr_net_sentiment'] = remarks_lr_probs[:, 2] - remarks_lr_probs[:, 0]
#Sentiment Scoring - Q&A - LR
P_summary['qa_lr_label'] = lr.predict(X_qa_vec)
qa_lr_probs = lr.predict_proba(X_qa_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_summary['qa_lr_net_sentiment'] = qa_lr_probs[:, 2] - qa_lr_probs[:, 0]

#Logistic Regression Sentiment Classification - Granular
#Vectorization already done so only sentiment scoring will be executed

#Sentiment Scoring - Text - LR
P_granular['txt_lr_label'] = lr.predict(X_text_vec)
text_lr_probs = lr.predict_proba(X_text_vec)
#Calculating continuous Net Probability Score (Pos - Neg)
P_granular['txt_lr_net_sentiment'] = text_lr_probs[:, 2] - text_lr_probs[:, 0]

#Saving Processed Data
output_path1 = '../Data/Transcripts/Summary_SentimentScore.csv'
output_path2 = '../Data/Transcripts/Granular_SentimentScore.csv'
P_summary.to_csv(output_path1, index=False)
P_granular.to_csv(output_path2, index=False)