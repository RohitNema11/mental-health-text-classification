# Mental Health Text Classification

A multiclass NLP classifier that detects 7 mental health conditions from text statements using TF-IDF vectorization and machine learning.

## Problem Statement
Mental health conditions are often expressed through language. This project builds a text classification model to automatically identify mental health conditions from user-written statements, which can aid in early detection and support systems.

## Dataset
**Source:** [Sentiment Analysis for Mental Health - Kaggle](https://www.kaggle.com/datasets/suchintika/sentiment-analysis-for-mental-health)  

> Download the dataset from Kaggle and place `Combined_Data.csv` in the root directory before running.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression, Multinomial Naive Bayes)
- NLTK (stopword removal)
- Matplotlib, Seaborn

## Methodology
1. **Preprocessing** — lowercasing, URL removal, punctuation removal, stopword filtering
2. **Feature Extraction** — TF-IDF vectorization with 10,000 features
3. **Modeling** — Logistic Regression vs Multinomial Naive Bayes
4. **Evaluation** — precision, recall, F1-score, confusion matrix across all 7 classes

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 76% | 0.72 |
| Multinomial Naive Bayes | 65% | 0.49 |

Logistic Regression outperforms Naive Bayes across all minority classes.
