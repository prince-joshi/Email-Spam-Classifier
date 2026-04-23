# Email Spam Classifier

A Natural Language Processing (NLP) project that classifies email/SMS messages as Spam or Ham (Not Spam) using machine learning.

---

## About the Project

This project builds a text classification model using an NLP pipeline. Raw messages are cleaned, preprocessed, and converted into numerical features using TF-IDF Vectorization. Two models were trained and compared — Naive Bayes and Logistic Regression. Logistic Regression performed better and was selected as the final model, deployed as an interactive Streamlit web app.

---

## Dataset

- **File:** `email_spam_data.csv`
- **Records:** 5572 messages
- **Columns:** Message, Label (spam/ham)
- **Target:** `label` (1 = Spam, 0 = Ham)

---

## NLP Pipeline

1. Lowercase conversion
2. Punctuation removal
3. Tokenization
4. Stopword removal
5. Stemming (PorterStemmer)
6. TF-IDF Vectorization

---

## Workflow

1. Loaded and explored the dataset
2. Visualized Spam vs Ham class distribution
3. Applied full NLP preprocessing pipeline
4. Split data into 80% train and 20% test
5. Vectorized text using TF-IDF
6. Trained and compared Naive Bayes and Logistic Regression
7. Evaluated using Accuracy, Classification Report, and Confusion Matrix heatmap
8. Saved the best model and vectorizer using joblib
9. Built a Streamlit app for real-time spam detection

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- NLTK (PorterStemmer, Stopwords)
- Streamlit
- Joblib

---

## Results

### Naive Bayes
| Metric | Value |
|--------|-------|
| Accuracy | 96.7% |
| Precision (Spam) | 1.00 |
| Recall (Spam) | 0.75 |
| F1-Score (Spam) | 0.86 |

### Logistic Regression (Final Model)
| Metric | Value |
|--------|-------|
| Accuracy | 97.8% |
| Precision (Spam) | 0.91 |
| Recall (Spam) | 0.93 |
| F1-Score (Spam) | 0.92 |

**Logistic Regression outperformed Naive Bayes — better recall on Spam class (93% vs 75%) means fewer spam messages go undetected. Selected as the final model.**

---

## Files

| File | Description |
|------|-------------|
| `email_spam_code.ipynb` | Main notebook — NLP pipeline, model training, evaluation |
| `email_spam_data.csv` | Dataset |
| `spam_classifier_model.pkl` | Saved Logistic Regression model |
| `vectorizer.pkl` | Saved TF-IDF Vectorizer |
| `spam_classifier_app.py` | Streamlit web app for real-time spam detection |

---

*Developed by Prince Joshi | Aspiring Data Analyst*
