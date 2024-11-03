import streamlit as st
from transformers import pipeline

sentiment_analysis = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

st.title("Sentiment Analysis")

with st.form("sentiment-analysis-form"):
    query = st.text_input("Enter your content for sentiment analysis")
    submitted = st.form_submit_button("Analyse Sentiment")

if submitted:
    if not query:
        st.error("Please enter your content for sentiment analysis")
    else:
        result = sentiment_analysis(query)
        
        if result[0][0]["score"]>0.5:
            st.success(result[0][0]["label"].upper())
            accuracy = result[0][0]["score"]
            st.write(f"Accuracy: {accuracy:.2f}")
        elif result[0][1]["score"]>0.5:
            st.write(result[0][1]["label"].upper())
            accuracy = result[0][1]["score"]
            st.write(f"Accuracy: {accuracy:.2f}")
        else:
            st.error(result[0][2]["label"].upper())
            accuracy = result[0][2]["score"]
            st.write(f"Accuracy: {accuracy:.2f}")