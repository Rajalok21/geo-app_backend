from flask import Flask, request, jsonify
from utils import get_data_from_query

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Geo-Backend is Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        result = get_data_from_query(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
