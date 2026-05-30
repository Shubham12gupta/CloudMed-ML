from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

from database import get_connection

app = Flask(__name__)
CORS(app)

MEDICAL_HISTORY_URL = os.getenv(
    "MEDICAL_HISTORY_URL",
    "http://43.204.191.63:5002"
)

CONSULTATION_URL = os.getenv(
    "CONSULTATION_URL",
    "http://43.204.191.63:5003"
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "backend-api"
    })


# ---------------------------------------------------
# MEDICAL HISTORY
# ---------------------------------------------------

@app.route("/api/medical-history/<tab>", methods=["POST"])
def medical_history(tab):

    allowed_tabs = [
        "medical_problems",
        "family_history",
        "risk_factors",
        "lifestyle",
        "allergies"
    ]

    if tab not in allowed_tabs:
        return jsonify({
            "error": f"Invalid tab. Must be one of {allowed_tabs}"
        }), 400

    payload = request.get_json()

    ml_response = requests.post(
        f"{MEDICAL_HISTORY_URL}/api/predict/{tab}",
        json=payload,
        timeout=120
    )

    prediction = ml_response.json()

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO prediction_results
            (prediction_type, result_json)
            VALUES (%s, %s)
            """,
            (
                tab,
                json.dumps(prediction)
            )
        )

        conn.commit()

        cur.close()
        conn.close()

    except Exception as e:
        print(f"DB Error: {e}")

    return jsonify(prediction)


# ---------------------------------------------------
# CONSULTATION
# ---------------------------------------------------

@app.route("/api/consultation/<tab>", methods=["POST"])
def consultation(tab):

    allowed_tabs = [
        "complaints",
        "findings",
        "diagnosis"
    ]

    if tab not in allowed_tabs:
        return jsonify({
            "error": f"Invalid tab. Must be one of {allowed_tabs}"
        }), 400

    payload = request.get_json()

    ml_response = requests.post(
        f"{CONSULTATION_URL}/api/predict/{tab}",
        json=payload,
        timeout=120
    )

    prediction = ml_response.json()

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO prediction_results
            (prediction_type, result_json)
            VALUES (%s, %s)
            """,
            (
                tab,
                json.dumps(prediction)
            )
        )

        conn.commit()

        cur.close()
        conn.close()

    except Exception as e:
        print(f"DB Error: {e}")

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000
    )
