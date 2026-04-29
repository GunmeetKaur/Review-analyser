"""
Customer Review Analysis Dashboard
Analyzes customer reviews from a CSV file using NLP.
"""

from collections import Counter

import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud

st.set_page_config(page_title="Customer Review Analysis", page_icon="📊", layout="wide")

def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:   return "Positive"
    elif score < 0: return "Negative"
    else:           return "Neutral"

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

df = df.rename(columns={"review text": "review_text"}).dropna(subset=["review_text"]).copy()
df["review_text"] = df["review_text"].astype(str).str.lower().str.strip()

# ── Sentiment ─────────────────────────────────────────────────────────────────
with st.spinner("Running sentiment analysis…"):
    df["sentiment"] = df["review_text"].apply(get_sentiment)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("📌 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", f"{len(df):,}")
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
st.dataframe(filtered[["review_text", "sentiment"]], use_container_width=True)

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

# ── Top words ─────────────────────────────────────────────────────────────────
st.subheader("🔤 Top 10 Words")
words = combined.split()
stopwords = {"the","a","an","and","or","but","in","on","at","to","for",
             "of","with","is","it","i","my","was","this","that","they",
             "have","had","not","be","as","are","we","so","me","he","she"}
words = [w for w in words if w not in stopwords and len(w) > 2]
top_words = pd.DataFrame(Counter(words).most_common(10), columns=["Word", "Count"])
st.dataframe(top_words, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.download_button(
    label="⬇️ Download Processed Data",
    data=filtered.to_csv(index=False),
    file_name="processed_reviews.csv",
    mime="text/csv",
)
