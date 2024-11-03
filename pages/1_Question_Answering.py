import streamlit as st 
from transformers import pipeline
import time
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder


st.title("Question & Answering")

ai_answer = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")



with st.form("question-answer-form"):
    content = st.text_area("Type your paragraph here...")
    query = st.text_input("What do you want to ask about the paragraph?")
    submitted = st.form_submit_button("Query")

if submitted:
    if not content or not query:
        st.error("Please enter a paragraph and ask a question")
    else:
        paragraphs = []
        for paragraph in content.replace("\r\n", "\n").split("\n\n"):
            if len(paragraph.strip()) > 0:
                paragraphs.append(sent_tokenize(paragraph.strip()))

        # We combine up to 3 sentences into a passage. You can choose smaller or larger values for window_size
        # Smaller value: Context from other sentences might get lost
        # Lager values: More context from the paragraph remains, but results are longer
        window_size = 3
        passages = []
        for paragraph in paragraphs:
            for start_idx in range(0, len(paragraph), window_size):
                end_idx = min(start_idx + window_size, len(paragraph))
                passages.append(" ".join(paragraph[start_idx:end_idx]))
        start_time = time.time()

        # Concatenate the query and all passages and predict the scores for the pairs [query, passage]
        model_inputs = [[query, passage] for passage in passages]
        scores = ai_answer.predict(model_inputs)

        # Sort the scores in decreasing order
        results = [{"input": inp, "score": score} for inp, score in zip(model_inputs, scores)]
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.write("Query:", query)
        # st.write(f"Search took {time.time() - start_time:.2f} seconds")
        
        value = results[0]["score"]
        if value > 0.5:
            st.write(results[0]["input"][1])
            st.write(f"Accuracy: {value:.2f}")
        else:
            st.error("I'm sorry, but that topic is outside the scope of my knowledge.")