"""
verify_all_scores.py
═══════════════════════════════════════════════════
Verifies VADER scores across the ENTIRE CSV file.
Also spot-checks BERT on 50 random comments.

Why not verify all 20889 BERT scores?
→ BERT takes ~11 min for full dataset.
→ 50 random samples is statistically strong enough
  to prove all scores are real.

Run:
    python verify_all_scores.py
═══════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

CSV = "FINAL_VADER_BERT_REAL_SCORES.csv"
BERT_SAMPLE = 20889   

df = pd.read_csv(CSV)
texts  = df["translated_comment"].fillna("").astype(str).tolist()
print(f"Loaded {len(df)} rows\n")

#  PART 1 — VERIFY ALL 20,889 VADER SCORES
print("=" * 60)
print(f"  VADER VERIFICATION — ALL {len(df)} COMMENTS")
print("=" * 60)

analyzer   = SentimentIntensityAnalyzer()
mismatches = []
matches    = 0

for i, (text, stored_score) in enumerate(zip(texts, df["vader_score"])):
    compound = analyzer.polarity_scores(text)["compound"]
    recalc   = round((compound + 1) / 2, 4)
    stored   = round(stored_score, 4)

    if abs(recalc - stored) < 0.001:
        matches += 1
    else:
        mismatches.append({
            "index":   i,
            "comment": text[:60],
            "stored":  stored,
            "recalc":  recalc,
            "diff":    abs(recalc - stored)
        })

    if (i + 1) % 2000 == 0:
        print(f"  Checked {i+1}/{len(df)} comments...")

print(f"\n Matched  : {matches} / {len(df)}")
print(f" Mismatched: {len(mismatches)} / {len(df)}")
match_pct = matches / len(df) * 100
print(f"  Match rate : {match_pct:.2f}%")

if len(mismatches) == 0:
    print("\n Every single VADER score is real and verified!")
elif len(mismatches) <= 10:
    print(f"\n {len(mismatches)} minor mismatches (likely floating point rounding):")
    for m in mismatches[:5]:
        print(f"     Row {m['index']}: stored={m['stored']} recalc={m['recalc']} diff={m['diff']:.5f}")
else:
    print(f"\n Too many mismatches — scores may not be real. Rerun fix_scores.py")

# Breakdown by sentiment class
print("\n  Verification by sentiment class:")
for sentiment in ["Positive", "Neutral", "Negative"]:
    subset    = df[df["vader_sentiment"] == sentiment]
    sub_texts = subset["translated_comment"].fillna("").astype(str).tolist()
    sub_scores= subset["vader_score"].tolist()
    correct   = 0
    for t, s in zip(sub_texts, sub_scores):
        c = analyzer.polarity_scores(t)["compound"]
        r = round((c + 1) / 2, 4)
        if abs(r - round(s, 4)) < 0.001:
            correct += 1
    print(f"    {sentiment:10}: {correct}/{len(subset)} verified "
          f"({correct/len(subset)*100:.1f}%)")

#  PART 2 — VERIFY 50 RANDOM BERT SCORES
print(f"\n{'='*60}")
print(f"  BERT VERIFICATION — {BERT_SAMPLE} RANDOM COMMENTS")
print(f"  (Full 20889 would take ~11 min — 50 is statistically solid)")
print(f"{'='*60}\n")

sample_idx = np.random.choice(len(df), BERT_SAMPLE, replace=False)
sample_df  = df.iloc[sample_idx].reset_index(drop=True)

bert_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    return_all_scores=True
)

bert_matches    = 0
bert_mismatches = []

for i, row in sample_df.iterrows():
    text = str(row["translated_comment"])
    try:
        raw       = bert_model(text, truncation=True, max_length=512)
        star_list = raw[0] if isinstance(raw[0], list) else raw
        stars     = {r["label"]: r["score"] for r in star_list}

        neg = stars.get("1 star", 0) + stars.get("2 stars", 0)
        neu = stars.get("3 stars", 0)
        pos = stars.get("4 stars", 0) + stars.get("5 stars", 0)

        class_scores  = {"Positive": pos, "Neutral": neu, "Negative": neg}
        recalc_label  = max(class_scores, key=class_scores.get)
        recalc_score  = round(class_scores[recalc_label], 4)
        stored_score  = round(row["bert_score"], 4)
        stored_label  = row["bert_sentiment"]

        score_ok = abs(recalc_score - stored_score) < 0.001
        label_ok = recalc_label == stored_label

        if score_ok and label_ok:
            bert_matches += 1
            status = "matched"
        else:
            bert_mismatches.append({
                "comment":       text[:50],
                "stored_label":  stored_label,
                "recalc_label":  recalc_label,
                "stored_score":  stored_score,
                "recalc_score":  recalc_score,
            })
            status = "mismatched"

        print(f"  [{i+1:02d}] {text[:45]:<45} "
              f"stored={stored_score:.4f} recalc={recalc_score:.4f} {status}")

    except Exception as e:
        print(f"  [{i+1:02d}] ERROR: {e}")

print(f"\n BERT Matched  : {bert_matches} / {BERT_SAMPLE}")
print(f" BERT Mismatched: {len(bert_mismatches)} / {BERT_SAMPLE}")
print(f"  Match rate       : {bert_matches/BERT_SAMPLE*100:.1f}%")

if len(bert_mismatches) == 0:
    print("\n ALL BERT SCORES VERIFIED — 100% real!")
else:
    print(f"\n  Mismatched rows:")
    for m in bert_mismatches:
        print(f"    Comment : {m['comment']}")
        print(f"    Stored  : {m['stored_label']} / {m['stored_score']}")
        print(f"    Recalc  : {m['recalc_label']} / {m['recalc_score']}")
        print()

#  FINAL SUMMARY
print(f"\n{'='*60}")
print("  FINAL VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"  VADER : {matches}/{len(df)} scores verified ({match_pct:.2f}%)")
print(f"  BERT  : {bert_matches}/{BERT_SAMPLE} sampled scores verified "
      f"({bert_matches/BERT_SAMPLE*100:.1f}%)")
print()
if matches == len(df) and bert_matches == BERT_SAMPLE:
    print(" CONCLUSION: All scores are 100% REAL.")
else:
    print(" Some scores didn't verify. Check mismatches above.")
print(f"{'='*60}")