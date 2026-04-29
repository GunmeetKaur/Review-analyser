"""
Customer Review Analysis Dashboard
Analyzes customer reviews from a CSV file using NLP.
"""

from collections import Counter

import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

st.set_page_config(page_title="Customer Review Analysis", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:   return "Positive"
    elif score < 0: return "Negative"
    else:           return "Neutral"


def extract_entities(text, nlp):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Review Analysis Dashboard")
st.caption("Upload a CSV file with a 'Review Text' column to get started.")

# ── File upload ───────────────────────────────────────────────────────────────
file = st.file_uploader("Upload CSV", type=["csv"])
if file is None:
    st.info("Waiting for a CSV file…")
    st.stop()

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(file, encoding="latin1", engine="python", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower()

if "review text" not in df.columns:
    st.error("CSV must contain a 'Review Text' column.")
    st.stop()

df = (
    df.rename(columns={"review text": "review_text"})
      .dropna(subset=["review_text"])
      .copy()
)
df["review_text"] = df["review_text"].astype(str).str.lower().str.strip()

# ── NLP ───────────────────────────────────────────────────────────────────────
nlp = load_model()

with st.spinner("Running analysis… this may take a moment on large files."):
    df["sentiment"] = df["review_text"].apply(get_sentiment)
    df["entities"]  = df["review_text"].apply(lambda x: extract_entities(x, nlp))

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("📌 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", len(df))
c2.metric("Positive",  (df["sentiment"] == "Positive").sum())
c3.metric("Negative",  (df["sentiment"] == "Negative").sum())
c4.metric("Neutral",   (df["sentiment"] == "Neutral").sum())

# ── Filters ───────────────────────────────────────────────────────────────────
st.subheader("🔍 Filter")
col_a, col_b = st.columns(2)
search           = col_a.text_input("Search keyword")
sentiment_filter = col_b.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])

filtered = df.copy()
if search:
    filtered = filtered[filtered["review_text"].str.contains(search, case=False, na=False)]
if sentiment_filter != "All":
    filtered = filtered[filtered["sentiment"] == sentiment_filter]

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("📄 Reviews")
st.dataframe(filtered[["review_text", "sentiment", "entities"]], use_container_width=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.subheader("📈 Sentiment Distribution")
st.bar_chart(filtered["sentiment"].value_counts())

# ── Word cloud ────────────────────────────────────────────────────────────────
st.subheader("☁️ Word Cloud")
combined = " ".join(filtered["review_text"])
if combined.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined)
    st.image(wc.to_array())
else:
    st.warning("No text to generate a word cloud.")

# ── Top entities ──────────────────────────────────────────────────────────────
st.subheader("🏷️ Top Named Entities")
all_entities = [e[0] for row in filtered["entities"] for e in row]
if all_entities:
    entity_df = pd.DataFrame(Counter(all_entities).most_common(10), columns=["Entity", "Count"])
    st.dataframe(entity_df, use_container_width=True)
else:
    st.info("No named entities found.")

# ── Download ──────────────────────────────────────────────────────────────────
st.download_button(
    label="⬇️ Download Processed Data",
    data=filtered.drop(columns=["entities"]).to_csv(index=False),
    file_name="processed_reviews.csv",
    mime="text/csv",
)
