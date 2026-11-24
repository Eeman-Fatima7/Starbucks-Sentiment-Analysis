# Starbucks Review Sentiment Analysis

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) to classify Starbucks customer reviews into Positive and Negative sentiments.

## Features
- Enter customer reviews.
- Automatically preprocess review text.
- Predict Positive or Negative sentiment.
- Display prediction results through an interactive Streamlit interface.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression / Multinomial Naive Bayes
- Streamlit

## Machine Learning Workflow
1. Exploratory Data Analysis
2. Data Cleaning and Preprocessing
3. Sentiment Label Creation
4. Text Vectorization using TF-IDF
5. Class Imbalance Handling using Random Oversampling
6. Model Training and Evaluation
7. Sentiment Prediction

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deployment
The application is deployed using Streamlit Community Cloud.

## Project Purpose
The purpose of this project is to analyse customer feedback and identify positive and negative experiences at Starbucks.