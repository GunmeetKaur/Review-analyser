import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import csv
import time
from streamlit_lottie import st_lottie
import requests

# Load spacy model with error handling
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("⚠️ Spacy model not found. Please check requirements.txt configuration.")
        st.stop()

nlp = load_spacy_model()

# Lottie animation loader
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# Animated title
placeholder = st.empty()
text = "📊 Customer Review Analysis Dashboard"
typed = ""

for char in text:
    typed += char
    placeholder.markdown(f"<h1 style='text-align:center'>{typed}</h1>", unsafe_allow_html=True)
    time.sleep(0.05)

# Load Lottie animation
lottie = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
if lottie:
    st_lottie(lottie, height=200)

# Styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Subtitle
st.markdown("Analyze real customer reviews using NLP 🚀")

# Upload CSV
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    # READ CSV (robust handling)
    df = pd.read_csv(
        file,
        encoding='latin1',
        engine='python',
        on_bad_lines='skip',
        sep=',',
        quoting=csv.QUOTE_NONE
    )

    # COLUMN CHECK
    if 'Review Text' not in df.columns:
        st.error("CSV must contain 'Review Text' column")
        st.stop()

    # Rename column
    df.rename(columns={'Review Text': 'review_text'}, inplace=True)

    # CLEANING
    df = df.dropna(subset=['review_text'])
    df['review_text'] = df['review_text'].astype(str)
    df['review_text'] = df['review_text'].str.lower()
    df['review_text'] = df['review_text'].str.strip()

    # PREVIEW
    st.subheader("📌 Data Preview")
    st.write(df.head())

    # SENTIMENT FUNCTION
    def get_sentiment(text):
        score = TextBlob(text).sentiment.polarity
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment'] = df['review_text'].apply(get_sentiment)
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(df))
    col2.metric("Positive", (df['sentiment'] == "Positive").sum())
    col3.metric("Negative", (df['sentiment'] == "Negative").sum())

    # ENTITY EXTRACTION
    def extract_entities(text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    df['entities'] = df['review_text'].apply(extract_entities)

    # SEARCH
    search = st.text_input("🔍 Search keyword")

    if search:
        df = df[df['review_text'].str.contains(search, case=False)]

    # FILTER
    sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"])

    if sentiment_filter != "All":
        df = df[df['sentiment'] == sentiment_filter]

    # PROCESSED DATA
    st.subheader("📊 Processed Data")
    st.write(df)

    # SENTIMENT CHART
    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

    # WORDCLOUD
    st.subheader("☁️ Word Cloud")
    text = " ".join(df['review_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wordcloud.to_array())

    # TOP ENTITIES
    st.subheader("🏷️ Top Entities")
    all_entities = []
    for ents in df['entities']:
        for e in ents:
            all_entities.append(e[0])

    entity_counts = Counter(all_entities)
    st.write(entity_counts.most_common(10))

    # DOWNLOAD
    st.download_button(
        "⬇️ Download Processed Data",
        df.to_csv(index=False),
        "processed_reviews.csv"
    )
