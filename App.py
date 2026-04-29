"""
Customer Review Analysis Dashboard
------------------------------------
Analyzes customer reviews from a CSV file.
Features: sentiment analysis, named entity recognition, word cloud, keyword search.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud


# ── App config (must be the first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📊",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_nlp_model():
    """Download (if needed) and load the spaCy English model once."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Use sys.executable to ensure we call the same Python that is running
        # this app — important on Streamlit Cloud where 'python' may not be in PATH
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


def get_sentiment(text: str) -> str:
    """Return Positive / Negative / Neutral using TextBlob polarity score."""
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"


def extract_entities(text: str, nlp) -> list[tuple[str, str]]:
    """Return a list of (entity_text, entity_label) pairs using spaCy NER."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def load_csv(file) -> pd.DataFrame | None:
    """
    Read uploaded CSV with common encoding/formatting issues handled.
    Returns a DataFrame, or None if the required column is missing.
    """
    try:
        df = pd.read_csv(file, encoding="latin1", engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

    if "Review Text" not in df.columns:
        st.error("CSV must contain a column named 'Review Text'.")
        return None

    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Rename column, drop nulls, and normalise text."""
    df = df.rename(columns={"Review Text": "review_text"}).copy()
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.lower().str.strip()
    return df


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    # Page header
    st.title("📊 Customer Review Analysis Dashboard")
    st.caption("Upload a CSV file with a 'Review Text' column to get started.")

    # Load model with spinner — first load downloads the model (~30s on cloud)
    with st.spinner("Loading NLP model… (first load may take ~30 seconds)"):
        nlp = load_nlp_model()

    # File upload
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.info("Waiting for a CSV file…")
        return

    # Load & clean
    df = load_csv(file)
    if df is None:
        return

    df = clean_reviews(df)

    # Run NLP
    df["sentiment"] = df["review_text"].apply(get_sentiment)
    df["entities"] = df["review_text"].apply(lambda t: extract_entities(t, nlp))

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("📌 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(df))
    col2.metric("Positive", (df["sentiment"] == "Positive").sum())
    col3.metric("Negative", (df["sentiment"] == "Negative").sum())
    col4.metric("Neutral",  (df["sentiment"] == "Neutral").sum())

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader("🔍 Filter Reviews")
    col_a, col_b = st.columns(2)

    with col_a:
        search = st.text_input("Search by keyword")
    with col_b:
        sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"])

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["review_text"].str.contains(search, case=False, na=False)]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    # ── Data table ────────────────────────────────────────────────────────────
    st.subheader("📄 Reviews")
    st.dataframe(filtered[["review_text", "sentiment", "entities"]], use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(filtered["sentiment"].value_counts())

    # ── Word cloud ────────────────────────────────────────────────────────────
    st.subheader("☁️ Word Cloud")
    combined_text = " ".join(filtered["review_text"])
    if combined_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        st.image(wc.to_array())
    else:
        st.warning("No text available to generate a word cloud.")

    # ── Top named entities ────────────────────────────────────────────────────
    st.subheader("🏷️ Top Named Entities")
    all_entity_texts = [
        entity_text
        for entity_list in filtered["entities"]
        for entity_text, _ in entity_list
    ]
    if all_entity_texts:
        top_entities = Counter(all_entity_texts).most_common(10)
        entity_df = pd.DataFrame(top_entities, columns=["Entity", "Count"])
        st.dataframe(entity_df, use_container_width=True)
    else:
        st.info("No named entities found in the filtered reviews.")

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Processed Data",
        data=filtered.drop(columns=["entities"]).to_csv(index=False),
        file_name="processed_reviews.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
