import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.header("📊 App Info")
    st.markdown("**Developer:** [Vedant Jadhav](https://www.linkedin.com/in/vedant-jadhav-566533273?)")
    st.markdown("**Tech Stack:** Python, Streamlit, Scikit-learn")
    st.markdown("**Dataset:** SMS Spam Collection")
    st.markdown("**GitHub:** [View Source](https://github.com/VedantJadhav701/)")

# --- Load & Train Model ---
@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
    df = df.rename(columns={"v1": "Category", "v2": "Message"})
    df = df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    X = df['Message']
    y = df['Spam']

    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X, y)
    return clf, df

model, df = train_model()

# --- App Title ---
st.markdown("""
    <h2 style='text-align: center; color: #0077cc;'>📧 Email Spam Detection App</h2>
    <p style='text-align: center;'>Enter your email text below and let the AI classify it as <b>Spam</b> or <b>Not Spam</b>.</p>
""", unsafe_allow_html=True)

# --- Input Box ---
email_input = st.text_area("✉️ Paste your email content here:", height=200)

# --- Predict Button ---
if st.button("🚀 Detect"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        proba = model.predict_proba([email_input])[0][1]
        prediction = model.predict([email_input])[0]
        st.write(f"Confidence: **{proba*100:.2f}%**")

        st.progress(proba)

        if prediction == 1:
            st.error("🚫 This email is likely **SPAM**.")
        else:
            st.success("✅ This email is **NOT SPAM**.")

# --- WordCloud Visualization ---
with st.expander("☁️ Show Spam WordCloud"):
    spam_words = " ".join(df[df['Spam'] == 1]['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
    st.image(wordcloud.to_array(), use_container_width=True)

# --- Classification Report ---
with st.expander("📋 Show Model Performance"):
    y = df['Spam']
    y_pred = model.predict(df['Message'])
    report = classification_report(y, y_pred, output_dict=True)
    st.json(report)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center;'>
Made by <a href='https://www.linkedin.com/in/vedant-jadhav-566533273?' target='_blank' style='text-decoration: none; color: #0077b5; font-weight: bold;'>Vedant Jadhav</a> |
<a href='https://github.com/VedantJadhav701/' target='_blank'>GitHub Repo</a>
</p>
""", unsafe_allow_html=True)
