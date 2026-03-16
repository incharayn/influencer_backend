import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load translated dataset
df = pd.read_csv("translated_instagram_comments.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sentiment function
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    compound = score['compound']
    
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply on translated text
df["vader_sentiment"] = df["translated_comment"].apply(get_sentiment)

print("Sentiment Analysis Completed")
print(df[["translated_comment", "vader_sentiment"]].head())

# Count sentiments
print("\nSentiment Distribution:")
print(df["vader_sentiment"].value_counts())

# Plot
df["vader_sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Save file
df.to_csv("instagram_comments_with_vader.csv", index=False)
print("File saved as instagram_comments_with_vader.csv")