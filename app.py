
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------
# 1. Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Starbucks Review Sentiment",
    page_icon="☕",
    layout="centered"
)

# ----------------------------------
# 2. Download NLTK Resources
# ----------------------------------

@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    return (
        set(stopwords.words("english")) - {"not", "no", "nor", "never"},
        WordNetLemmatizer()
    )

stop_words, lemmatizer = setup_nltk()

# ----------------------------------
# 3. Load Trained Model
# ----------------------------------

@st.cache_resource
def load_model():
    artifact = joblib.load("starbucks_sentiment_model.pkl")
    return artifact["model"], artifact["tfidf"]

model, tfidf = load_model()

# ----------------------------------
# 4. Text Cleaning Function
# ----------------------------------

def clean_review(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ----------------------------------
# 5. Application Interface
# ----------------------------------

st.title("☕ Starbucks Review Sentiment Analysis")

st.write(
    "Enter a Starbucks customer review below "
    "to predict whether the review is Positive or Negative."
)

st.divider()

review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: I love the coffee and friendly staff...",
    height=150
)

# ----------------------------------
# 6. Prediction
# ----------------------------------

if st.button("Analyze Review", type="primary"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_review(review)

        if not cleaned_review.strip():
            st.warning("Please enter a meaningful review.")

        else:
            review_vector = tfidf.transform([cleaned_review])
            prediction = model.predict(review_vector)[0]

            st.subheader("Prediction Result")

            if prediction == "Positive":
                st.success("😊 Positive Review")

            else:
                st.error("☹️ Negative Review")

            with st.expander("View cleaned review"):
                st.write(cleaned_review)

st.divider()

st.caption(
    "Developed using TF-IDF and machine learning "
    "for Starbucks customer review sentiment analysis."
)
