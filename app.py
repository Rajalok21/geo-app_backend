from flask import Flask, request, jsonify
from utils import process_query

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ GeoIntelligence API is running!",
        "endpoints": {
            "POST /analyze": "Send a query like 'find kutcha houses near MG Road with hospitals'"
        }
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        result = process_query(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
