import streamlit as st
import pickle
import pandas as pd
import os

# -----------------------------
# Safe File Loading
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sentiment_model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
label_encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")
condition_model_path = os.path.join(BASE_DIR, "condition_model.pkl")

# Load models
sentiment_model = pickle.load(open(sentiment_model_path, "rb"))
tfidf = pickle.load(open(tfidf_path, "rb"))
label_encoder = pickle.load(open(label_encoder_path, "rb"))
condition_model = pickle.load(open(condition_model_path, "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Drug Review Analysis", layout="wide")

st.title("Patient Condition Classification using Drug Reviews")
st.write("Enter a drug review to predict sentiment and possible medical condition.")

user_input = st.text_area("Enter Review Text:", height=250)

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform text
        vector = tfidf.transform([user_input])

        # Sentiment prediction
        sentiment_pred = sentiment_model.predict(vector)[0]
        sentiment_prob = sentiment_model.predict_proba(vector)[0]
        sentiment_text = sentiment_labels[sentiment_pred]

        # Condition prediction
        condition_pred = condition_model.predict(vector)[0]
        condition_label = label_encoder.inverse_transform([condition_pred])[0]

        # -----------------------------
        # Display Results
        # -----------------------------

        st.subheader("Prediction Results")

        st.success(f"Predicted Sentiment: {sentiment_text}")
        st.info(f"Most Likely Condition: {condition_label}")

        result_table = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability (%)": [
                round(sentiment_prob[0] * 100, 2),
                round(sentiment_prob[1] * 100, 2),
                round(sentiment_prob[2] * 100, 2)
            ]
        })

        st.subheader("Sentiment Probability Breakdown")
        st.table(result_table)
