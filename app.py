from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Backend Running"})

@app.route("/campaign", methods=["POST"])
def campaign():
    data = request.get_json()

    print("Received Campaign:", data)

    return jsonify({
        "status": "success",
        "message": "Campaign received",
        "data": data
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
