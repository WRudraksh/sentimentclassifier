import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('deepcsat_RReview.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

# Tokenization (same as in your Colab code)
def simple_tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

# Stopword removal (same as training)
def remove_stopwords(token):
    return [word for word in token if word.lower() not in stop_words]

# Final prediction function
def predict_review(text):
    tokens = simple_tokenize(text)
    tokens_no_stopwords = remove_stopwords(tokens)
    text_cleaned = ' '.join(tokens_no_stopwords)
    X_tfidf = tfidf_vectorizer.transform([text_cleaned])
    y_pred = model.predict(X_tfidf)
    return y_pred[0]

# Streamlit Interface
st.title("Restaurant Review Sentiment Classifier")

review_input = st.text_area("Enter a customer review below:")

if st.button("Predict"):
    if not review_input.strip():
        st.warning("Please enter a review.")
    else:
        sentiment = predict_review(review_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
