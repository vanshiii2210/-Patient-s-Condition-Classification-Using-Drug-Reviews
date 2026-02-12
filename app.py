import streamlit as st
import pickle
import pandas as pd

# Load models
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Page config
st.set_page_config(page_title="Drug Review Sentiment Analysis", layout="wide")

st.title("Drug Review Sentiment Analysis App")

# Input box
user_input = st.text_area("Enter your review text here:")

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = tfidf.transform([user_input])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        sentiment = label_map[prediction]

        st.subheader(f"Predicted Sentiment: {sentiment}")

        results = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability (%)": [round(probability[0]*100,2),
                                round(probability[1]*100,2),
                                round(probability[2]*100,2)]
        })

        st.table(results)
