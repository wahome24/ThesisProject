Topic : A Comparative Study of Traditional and LLM-Based Sentiment Analysis Techniques on Financial News and Earnings Call Transcripts for S&P 500 Stock Movement Prediction

The study aimed to evaluate the relative predictive power of traditional machine learning models (SVM and Logistic Regression) versus contextual large language models (FinBERT and GPT) across two distinct textual media sources: continuous financial news and quarterly corporate Earnings Call Transcripts (ECTs). Predictions were evaluated on an unseen, out-of-time test set from 2023, considering 1-day (T+1) and 5-day (T+5) forward movement horizons. This analysis yielded four primary conclusions:
The superiority of an ML or LLM framework depends on the time horizon and the model architecture.
News is quickly absorbed by the market, leaving no residual sentiment of sufficient magnitude to predict market movements, providing evidence in support of the Semi-Strong Form Efficient Market Hypothesis (EMH).
ECT predictive signal is undetectable on the T+1 horizon due to short-term volatility, necessitating a five-day post-announcement window (T+5) to become apparent in equity returns. This finding supports the established literature on Post-Earnings Announcement Drift (PEAD).
Combining news with ECT introduces noise, resulting in lower performance compared to using individual text streams. 
