import os
from flask import Flask, request, jsonify
from utils import process_query

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ API is live!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400
    result = process_query(query)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render injects PORT env variable
    app.run(host="0.0.0.0", port=port, debug=True)  # ✅ Bind to 0.0.0.0
