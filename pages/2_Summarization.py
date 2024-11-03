import streamlit as st
from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

st.title("Summarize the content")



with st.form('summarization-form'):
    query = st.text_area("Type your paragraph here...")
    submitted = st.form_submit_button("Summarize the paragraph")

if submitted:
    if not query:
        st.error("Paragraph is empty")
    else:
        text_to_summarize = query
        inputs = tokenizer(text_to_summarize,max_length=1024,return_tensors="pt")

        summary_ids = model.generate(inputs["input_ids"],num_beams=2, min_length=0, max_length=50)
        result = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        st.write(result)