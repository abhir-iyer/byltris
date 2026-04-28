import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/cfpb_complaints.parquet"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Load narratives ───────────────────────────────────────────────────────

def load_narratives():
    df = pd.read_parquet(DATA_PATH)

    # keep rows with actual narrative text
    df = df[df["consumer_complaint_narrative"].notna()].copy()
    df = df[df["consumer_complaint_narrative"].str.len() > 50].copy()

    # normalize company names
    def normalize(name):
        n = str(name).lower()
        if any(x in n for x in ["truist", "suntrust", "bb&t"]): return "Truist"
        if "bank of america" in n: return "Bank of America"
        if "wells fargo" in n: return "Wells Fargo"
        if "jpmorgan" in n or "chase" in n: return "JPMorgan Chase"
        if "citibank" in n: return "Citibank"
        return "Other"

    df["institution"] = df["company"].apply(normalize)
    df["year"] = pd.to_datetime(df["date_received"]).dt.year
    df["text"] = df["consumer_complaint_narrative"].str.lower().str.strip()

    print(f"Narratives available: {len(df):,}")
    print(f"By institution:\n{df['institution'].value_counts()}")
    return df


# ── 2. BERTopic clustering ───────────────────────────────────────────────────

def run_bertopic(df, institution="Truist", n_docs=5000, n_topics=15):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    sub = df[df["institution"] == institution].copy()

    # sample for speed — BERTopic on full corpus is slow on CPU
    if len(sub) > n_docs:
        sub = sub.sample(n_docs, random_state=42)

    docs = sub["text"].tolist()
    print(f"\nRunning BERTopic on {len(docs):,} {institution} narratives...")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedder,
        nr_topics=n_topics,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(docs)
    sub = sub.copy()
    sub["topic"] = topics

    # get topic labels
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered topics:")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    return topic_model, sub, topic_info


# ── 3. Topic trend over time ─────────────────────────────────────────────────

def topic_trends(df_with_topics, topic_model):
    topic_info = topic_model.get_topic_info()
    topic_names = dict(zip(topic_info["Topic"], topic_info["Name"]))

    df_with_topics["topic_name"] = df_with_topics["topic"].map(topic_names)

    # complaints per topic per year
    trend = (
        df_with_topics[df_with_topics["topic"] != -1]
        .groupby(["year", "topic_name"])
        .size()
        .reset_index(name="count")
    )

    out_path = f"{OUT_DIR}/cfpb_topic_trends.parquet"
    trend.to_parquet(out_path, index=False)
    print(f"\nTopic trends saved to {out_path}")

    print("\nTop topics by total volume:")
    print(trend.groupby("topic_name")["count"].sum().sort_values(ascending=False).head(10))
    return trend


# ── 4. Keyword frequency analysis (fast alternative to BERTopic) ─────────────

def keyword_analysis(df, institution="Truist"):
    """
    Fast keyword frequency analysis on narratives.
    Identifies top complaint themes without GPU.
    """
    from collections import Counter
    import re

    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","was","are","were","been","be",
        "have","has","had","will","would","could","should","may","might",
        "i","my","me","we","our","they","their","it","its","this","that",
        "not","no","so","if","as","up","out","about","into","after",
        "bank","account","loan","payment","credit","customer","said",
        "told","called","received","sent","made","also","would","xxxx",
        "xx","xxx","co","inc","na","llc"
    }

    sub = df[df["institution"] == institution].copy()
    all_words = []

    for text in sub["text"].dropna():
        words = re.findall(r"\b[a-z]{4,}\b", str(text).lower())
        all_words.extend([w for w in words if w not in STOPWORDS])

    counter = Counter(all_words)
    top_words = pd.DataFrame(counter.most_common(50), columns=["word", "count"])

    print(f"\nTop 50 keywords in {institution} complaints:")
    print(top_words.to_string(index=False))

    out_path = f"{OUT_DIR}/cfpb_keywords_{institution.lower().replace(' ','_')}.parquet"
    top_words.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return top_words


# ── 5. Complaint velocity signal ─────────────────────────────────────────────

def complaint_velocity(df):
    """
    Flag rising complaint velocity by product × institution.
    A 2-sigma rise above 4-quarter rolling mean = emerging risk signal.
    """
    df["date"] = pd.to_datetime(df["date_received"])
    df["quarter"] = df["date"].dt.to_period("Q")

    vol = (
        df[df["institution"] != "Other"]
        .groupby(["institution", "product", "quarter"])
        .size()
        .reset_index(name="count")
        .sort_values(["institution", "product", "quarter"])
    )

    vol["rolling_mean"] = (
        vol.groupby(["institution", "product"])["count"]
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    vol["rolling_std"] = (
        vol.groupby(["institution", "product"])["count"]
        .transform(lambda x: x.rolling(4, min_periods=2).std())
    )
    vol["z_score"] = (vol["count"] - vol["rolling_mean"]) / vol["rolling_std"].replace(0, np.nan)
    vol["rising_flag"] = vol["z_score"] > 2.0

    flagged = vol[vol["rising_flag"]].sort_values("z_score", ascending=False)
    print(f"\nRising complaint velocity flags (z > 2.0):")
    print(flagged[["institution","product","quarter","count","z_score"]].head(20).to_string(index=False))

    out_path = f"{OUT_DIR}/cfpb_velocity_signals.parquet"
    vol.to_parquet(out_path, index=False)
    print(f"\nVelocity signals saved to {out_path}")
    return vol, flagged


# ── 6. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_narratives()

    # keyword analysis (fast, no GPU needed)
    kw = keyword_analysis(df, "Truist")
    kw_peers = keyword_analysis(df, "Bank of America")

    # complaint velocity signals
    vol, flagged = complaint_velocity(df)

    # BERTopic (comment out if too slow on your machine)
    print("\nRunning BERTopic (this takes 3-5 mins on CPU)...")
    try:
        topic_model, df_topics, topic_info = run_bertopic(df, "Truist", n_docs=3000)
        trends = topic_trends(df_topics, topic_model)
    except Exception as e:
        print(f"BERTopic skipped: {e}")
        print("Keyword analysis and velocity signals are sufficient for the paper.")

    print("\nDone.")
# save topic model
    import pickle
    nlp_artifacts = {
        "keywords_truist": kw,
        "keywords_bofa": kw_peers,
        "velocity_signals": vol,
    }
    try:
        topic_model.save(f"{MODEL_DIR}/bertopic_truist")
        print(f"BERTopic model saved to {MODEL_DIR}/bertopic_truist")
    except Exception as e:
        print(f"BERTopic save skipped: {e}")

    with open(f"{MODEL_DIR}/nlp_artifacts.pkl", "wb") as f:
        pickle.dump(nlp_artifacts, f)
    print(f"NLP artifacts saved to {MODEL_DIR}/nlp_artifacts.pkl")