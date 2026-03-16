from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# load dataset
data = pd.read_csv("influencer_master.csv")


@app.route("/influencers", methods=["GET"])
def get_influencers():
    influencers = data.head(10).to_dict(orient="records")
    return jsonify(influencers)


@app.route("/campaign", methods=["POST"])
def create_campaign():

    campaign = request.json

    print("Campaign received:", campaign)

    # here you could run ranking or sentiment logic

    return jsonify({
        "status": "success",
        "message": "Campaign processed"
    })


if __name__ == "__main__":
    app.run()