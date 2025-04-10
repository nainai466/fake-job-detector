
import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open('job_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_input(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("🕵️‍♂️ Fake Job Detector - Pakistan Edition")
st.markdown("Check if a job post is fake or real using Machine Learning.")

job_input = st.text_area("Paste the Job Ad text here:")

if st.button("Check Now"):
    cleaned = clean_input(job_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)

    if prediction[0] == 1:
        st.error("⚠️ Warning: This job ad looks **FAKE**. Please be careful.")
    else:
        st.success("✅ This job ad seems **REAL**.")
