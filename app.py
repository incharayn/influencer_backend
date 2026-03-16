from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset once
data = pd.read_csv("FINAL_VADER_BERT_WITH_SCORE.csv")

@app.route("/")
def home():
    return "Influencer Backend Running"

@app.route("/api/influencers")
def influencers():
    result = data.head(20).to_dict(orient="records")
    return jsonify(result)

if __name__ == "__main__":
    app.run()
