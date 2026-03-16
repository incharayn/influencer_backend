"""
PATH 2: Evaluate VADER using BERT as pseudo ground truth
Use this when you have NO manually labeled data.
BERT is treated as the reference model, VADER is evaluated against it.
This is a valid approach for exploratory/comparative NLP projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    accuracy_score, ConfusionMatrixDisplay
)

LABELS = ["Positive", "Neutral", "Negative"]

df = pd.read_csv("FINAL_VADER_BERT_REAL_SCORES.csv")
print(f"Total comments: {len(df)}")

y_bert  = df["bert_sentiment"]   
y_vader = df["vader_sentiment"]  

print(f"\n{'═'*55}")
print("  VADER evaluated against BERT (BERT = reference)")
print(f"{'═'*55}")

acc  = accuracy_score(y_bert, y_vader)
f1w  = f1_score(y_bert, y_vader, average="weighted", labels=LABELS, zero_division=0)
f1m  = f1_score(y_bert, y_vader, average="macro",    labels=LABELS, zero_division=0)
prec = precision_score(y_bert, y_vader, average="weighted", labels=LABELS, zero_division=0)
rec  = recall_score(y_bert, y_vader,    average="weighted", labels=LABELS, zero_division=0)

print(f"\n  Accuracy (Agreement)  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 Score (weighted)   : {f1w:.4f}")
print(f"  F1 Score (macro)      : {f1m:.4f}")
print(f"  Precision (weighted)  : {prec:.4f}")
print(f"  Recall (weighted)     : {rec:.4f}")

print(f"\n  Per-class breakdown (how well VADER matches BERT per label):")
print(classification_report(y_bert, y_vader, labels=LABELS, zero_division=0))

print(f"\n{'═'*55}")
print("  DISAGREEMENT ANALYSIS")
print(f"{'═'*55}")

disagree = df[df["vader_sentiment"] != df["bert_sentiment"]]
print(f"  Total disagreements: {len(disagree)} / {len(df)}  ({len(disagree)/len(df)*100:.1f}%)")
print(f"\n  VADER → BERT disagreement breakdown:")
print(disagree.groupby(["vader_sentiment", "bert_sentiment"]).size()
      .rename("count").reset_index().to_string(index=False))

print(f"\n  Sample of disagreeing comments:")
sample_dis = disagree[["translated_comment","vader_sentiment","bert_sentiment"]].sample(10, random_state=42)
for _, row in sample_dis.iterrows():
    print(f"\n  Comment : {str(row['translated_comment'])[:80]}")
    print(f"  VADER   : {row['vader_sentiment']}")
    print(f"  BERT    : {row['bert_sentiment']}")

print(f"\n{'═'*55}")
print("SENTIMENT DISTRIBUTION COMPARISON")
print(f"{'═'*55}")
dist = pd.DataFrame({
    "VADER": df["vader_sentiment"].value_counts(),
    "BERT":  df["bert_sentiment"].value_counts()
}).reindex(LABELS).fillna(0).astype(int)
dist["Difference"] = dist["BERT"] - dist["VADER"]
print(dist.to_string())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("VADER vs BERT Sentiment Evaluation\n(BERT used as reference)", 
             fontsize=14, fontweight="bold")
colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}

ax = axes[0][0]
vc = df["vader_sentiment"].value_counts().reindex(LABELS)
ax.bar(LABELS, vc.values, color=[colors_map[l] for l in LABELS], alpha=0.85)
ax.set_title("VADER Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(vc.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

ax = axes[0][1]
bc = df["bert_sentiment"].value_counts().reindex(LABELS)
ax.bar(LABELS, bc.values, color=[colors_map[l] for l in LABELS], alpha=0.85)
ax.set_title("BERT Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(bc.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

ax = axes[0][2]
agree_count = (df["vader_sentiment"] == df["bert_sentiment"]).sum()
ax.pie([agree_count, len(df)-agree_count],
       labels=[f"Agree\n{agree_count}", f"Disagree\n{len(df)-agree_count}"],
       colors=["#4CAF50", "#F44336"],
       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
ax.set_title("VADER vs BERT Agreement")

ax = axes[1][0]
cm = confusion_matrix(y_bert, y_vader, labels=LABELS)
disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix\n(BERT=true, VADER=pred)")
ax.set_xlabel("VADER (predicted)")
ax.set_ylabel("BERT (reference)")

ax = axes[1][1]
metric_names = ["Accuracy", "F1\n(weighted)", "F1\n(macro)", "Precision", "Recall"]
metric_values = [acc, f1w, f1m, prec, rec]
bars = ax.bar(metric_names, metric_values, color="#2196F3", alpha=0.85)
ax.set_ylim(0, 1.15)
ax.set_title("VADER Metrics\n(vs BERT as reference)")
ax.set_ylabel("Score")
ax.axhline(0.7, color="orange", linestyle="--", alpha=0.6, label="0.7 threshold")
ax.legend(fontsize=9)
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

ax = axes[1][2]
f1_per = f1_score(y_bert, y_vader, average=None, labels=LABELS, zero_division=0)
bar_colors = [colors_map[l] for l in LABELS]
bars2 = ax.bar(LABELS, f1_per, color=bar_colors, alpha=0.85)
ax.set_ylim(0, 1.15)
ax.set_title("F1 Score Per Sentiment Class\n(VADER vs BERT)")
ax.set_ylabel("F1 Score")
for bar, val in zip(bars2, f1_per):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("vader_vs_bert_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Chart saved: vader_vs_bert_evaluation.png")

print(f"\n{'═'*55}")
print("  FINAL SUMMARY")
print(f"{'═'*55}")
print(f"  Accuracy  : {acc:.4f}  → VADER agrees with BERT {acc*100:.1f}% of the time")
print(f"  F1(weighted) : {f1w:.4f}  → weighted average performance")
print(f"  Precision : {prec:.4f}  → when VADER predicts a class, how often it matches BERT")
print(f"  Recall    : {rec:.4f}  → how well VADER catches each class that BERT found")
