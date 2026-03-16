from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend Running"

@app.route("/campaign", methods=["POST"])
def campaign():
    data = request.get_json()

    campaign_name = data.get("campaignName")
    budget = data.get("campaignBudget")
    age_group = data.get("ageGroup")
    location = data.get("location")
    niche = data.get("niche")
    influencer_count = data.get("influencerCount")

    print("Received Campaign:", data)

    return jsonify({
        "status": "success",
        "message": "Campaign received",
        "campaign": campaign_name,
        "influencers_requested": influencer_count
    })
