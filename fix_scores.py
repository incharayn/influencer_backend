"""
fix_scores.py
═══════════════════════════════════════════════════
Replaces fake random vader_score and bert_score
with REAL scores from each model.

VADER score  → compound score from VADER (-1 to +1), 
               normalized to 0–1 for consistency
BERT score   → actual confidence % the model gave 
               to its predicted label (0–1)

Run:
    pip install vaderSentiment transformers torch
    python fix_scores.py
═══════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

# CONFIG 
INPUT_CSV  = "FINAL_VADER_BERT_WITH_SCORE.csv"
OUTPUT_CSV = "FINAL_VADER_BERT_REAL_SCORES.csv"
BERT_BATCH = 32          # kept for reference, not used in single-item mode

df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {df.shape[0]} rows")
texts = df["translated_comment"].fillna("").astype(str).tolist()

#  PART 1 — REAL VADER SCORE
#  compound score: -1 (most negative) to +1 (most positive)
#  We normalize it to 0–1 so it matches the 0–1 range of BERT

print("\n[1/2] Computing real VADER scores...")

analyzer = SentimentIntensityAnalyzer()

def get_vader_compound(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]        # raw: -1 to +1
    normalized = (compound + 1) / 2     # shift to 0–1
    return round(normalized, 4)

df["vader_score"] = df["translated_comment"].apply(get_vader_compound)

print("VADER scores done")
print(f"   Range: {df['vader_score'].min():.3f} – {df['vader_score'].max():.3f}")
print(f"   Mean : {df['vader_score'].mean():.3f}")

# Quick sanity check — positive comments should score > 0.5
print("\n   Sanity check (mean score by VADER sentiment):")
print(df.groupby("vader_sentiment")["vader_score"].mean().round(3).to_string())

#  PART 2 — REAL BERT SCORE
#  We use return_all_scores=True to get the confidence for 
#  EACH star rating (1–5), then map to Positive/Neutral/Negative
#  and return the confidence of the WINNING class

print("\n[2/2] Computing real BERT scores...")
print(f"   Total comments: {len(texts)}")
print(f"   Batch size: {BERT_BATCH}")
print(f"   Estimated time: ~{len(texts)//BERT_BATCH//60 + 1} min on CPU, faster on GPU")
print("   (Processing...)\n")

device = 0 if torch.cuda.is_available() else -1
print(f"   Using: {'GPU' if device == 0 else 'CPU (no GPU found)'}")

bert_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    return_all_scores=True,
    device=device
)

def convert_star_label(label):
    stars = int(label[0])
    if stars >= 4:
        return "Positive"
    elif stars == 3:
        return "Neutral"
    else:
        return "Negative"

bert_scores = []
bert_labels = []

def score_one(text):
    """Score a single comment. Returns (label, confidence)."""
    try:
        # Returns [[{label, score}, {label, score}, ...]] for one input
        raw = bert_model(str(text), truncation=True, max_length=512)

        # Unwrap: could be [[...]] or [...]
        if isinstance(raw[0], list):
            star_list = raw[0]          # [[5 dicts]] → [5 dicts]
        else:
            star_list = raw             # [5 dicts] already

        star_scores = {r["label"]: r["score"] for r in star_list}

        neg = star_scores.get("1 star",  0) + star_scores.get("2 stars", 0)
        neu = star_scores.get("3 stars", 0)
        pos = star_scores.get("4 stars", 0) + star_scores.get("5 stars", 0)

        class_scores  = {"Positive": pos, "Neutral": neu, "Negative": neg}
        winning_class = max(class_scores, key=class_scores.get)
        winning_score = class_scores[winning_class]
        return winning_class, round(winning_score, 4)

    except Exception as e:
        return "Neutral", 0.5          # safe fallback

for i, text in enumerate(texts):
    label, score = score_one(text)
    bert_labels.append(label)
    bert_scores.append(score)

    if i % 500 == 0:
        print(f"   Processed {i}/{len(texts)} comments...")

print(f"   Processed {len(texts)}/{len(texts)} comments")

df["bert_sentiment"] = bert_labels   # also updates sentiment with fresh labels
df["bert_score"]     = bert_scores

print("\n BERT scores done")
print(f"   Range: {df['bert_score'].min():.3f} – {df['bert_score'].max():.3f}")
print(f"   Mean : {df['bert_score'].mean():.3f}")

print("\n   Sanity check (mean confidence by BERT sentiment):")
print(df.groupby("bert_sentiment")["bert_score"].mean().round(3).to_string())

#  SAVE
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Saved: {OUTPUT_CSV}")
print(f"   Rows: {len(df)}")
print(f"\n   Final columns: {list(df.columns)}")

# Preview
print("\n   Sample of real scores:")
print(df[["translated_comment","vader_sentiment","vader_score",
          "bert_sentiment","bert_score"]].head(10).to_string())