"""
Customer Review Analysis Dashboard
"""

# ── Standard library ─────────────────────────────
import os
import sys
from collections import Counter

# ── Third-party ─────────────────────────────────
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

# ── App config ──────────────────────────────────
st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Load NLP model (cached) ─────────────────────
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

# ── Sentiment function ──────────────────────────
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"

# ── Entity extraction ───────────────────────────
def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# ── Load CSV ────────────────────────────────────
def load_csv(file):
    try:
        df = pd.read_csv(file, encoding="latin1", engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "review text" not in df.columns:
        st.error("CSV must contain 'Review Text' column")
        return None

    df.rename(columns={"review text": "review_text"}, inplace=True)
    return df

# ── Clean data ──────────────────────────────────
def clean_data(df):
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.lower().str.strip()
    return df

# ── MAIN APP ────────────────────────────────────
def main():

    st.title("📊 Customer Review Analysis Dashboard")
    st.caption("Upload a CSV file with a 'Review Text' column")

    # Upload file FIRST
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is None:
        st.info("Waiting for a CSV file…")
        return

    st.success(f"File uploaded: {file.name}")

    # Load model AFTER upload (fixes bug)
    with st.spinner("Loading NLP model..."):
        nlp = load_nlp_model()

    # Load data
    df = load_csv(file)
    if df is None:
        return

    df = clean_data(df)

    # Apply NLP
    df["sentiment"] = df["review_text"].apply(get_sentiment)
    df["entities"] = df["review_text"].apply(lambda x: extract_entities(x, nlp))

    # ── Metrics ───────────────────────────────
    st.subheader("📌 Summary")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(df))
    col2.metric("Positive", (df["sentiment"] == "Positive").sum())
    col3.metric("Negative", (df["sentiment"] == "Negative").sum())
    col4.metric("Neutral", (df["sentiment"] == "Neutral").sum())

    # ── Filters ───────────────────────────────
    st.subheader("🔍 Filter Reviews")
    search = st.text_input("Search keyword")
    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    filtered = df.copy()

    if search:
        filtered = filtered[filtered["review_text"].str.contains(search, case=False, na=False)]

    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    # ── Table ────────────────────────────────
    st.subheader("📄 Reviews")
    st.dataframe(filtered[["review_text", "sentiment", "entities"]])

    # ── Chart ────────────────────────────────
    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(filtered["sentiment"].value_counts())

    # ── WordCloud ────────────────────────────
    st.subheader("☁️ Word Cloud")
    text = " ".join(filtered["review_text"])

    if text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array())
    else:
        st.warning("No text for word cloud")

    # ── Entities ─────────────────────────────
    st.subheader("🏷️ Top Entities")

    all_entities = [
        entity_text
        for entity_list in filtered["entities"]
        for entity_text, _ in entity_list
    ]

    if all_entities:
        top_entities = Counter(all_entities).most_common(10)
        entity_df = pd.DataFrame(top_entities, columns=["Entity", "Count"])
        st.dataframe(entity_df)
    else:
        st.info("No entities found")

    # ── Download ─────────────────────────────
    st.download_button(
        "⬇️ Download Processed Data",
        filtered.drop(columns=["entities"]).to_csv(index=False),
        "processed_reviews.csv"
    )

# ── RUN ────────────────────────────────────────
if __name__ == "__main__":
    main()
