# app.py — Medical ML API (All 5 Tabs)

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import get_recommendations, TABS
import traceback

app = Flask(__name__)
CORS(app)

# ── Health check ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'healthy',
        'tabs'   : TABS,
        'message': 'Medical ML API running — all 5 tabs ready'
    })


# ── Generic predict endpoint (handles all 5 tabs) ────────────
@app.route('/api/predict/<tab_name>', methods=['POST'])
def predict(tab_name):
    """
    Single endpoint for all 5 tabs.

    URL:
        POST /api/predict/medical_problems
        POST /api/predict/family_history
        POST /api/predict/risk_factors
        POST /api/predict/lifestyle
        POST /api/predict/allergies

    Request body:
        {
            "patient": {
                "age"         : 52,
                "gender"      : "Male",
                "blood_group" : "B+",
                "season"      : "Winter",
                "has_history" : true
            },
            "known_tabs": {
                "medical_problems": ["Hypertension", "Diabetes"],
                "family_history"  : ["Father had Diabetes"]
            }
        }

    Response:
        {
            "tab"        : "risk_factors",
            "suggestions": [
                {"label": "High Stress Levels", "confidence": 62.3},
                {"label": "Physical Inactivity", "confidence": 58.1},
                ...
            ]
        }
    """

    # Validate tab name
    if tab_name not in TABS:
        return jsonify({
            'error': f'Invalid tab: {tab_name}. Must be one of {TABS}'
        }), 400

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON body sent'}), 400

        patient    = data.get('patient', {})
        known_tabs = data.get('known_tabs', {})

        # Validate required patient fields
        required = ['age', 'gender', 'blood_group', 'season']
        missing  = [f for f in required if f not in patient]
        if missing:
            return jsonify({
                'error'  : f'Missing patient fields: {missing}',
                'example': {
                    'age'         : 45,
                    'gender'      : 'Male',
                    'blood_group' : 'B+',
                    'season'      : 'Winter',
                    'has_history' : True
                }
            }), 400

        # Get recommendations
        results = get_recommendations(
            patient_profile = patient,
            known_tabs      = known_tabs,
            target_tab      = tab_name,
            top_n           = 10
        )

        return jsonify({
            'tab'        : tab_name,
            'suggestions': [
                {'label': label, 'confidence': conf}
                for label, conf in results
            ]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Individual tab endpoints (explicit, easier for backend) ──
@app.route('/api/predict/medical-problems', methods=['POST'])
def predict_medical():
    request._tab = 'medical_problems'
    return predict('medical_problems')

@app.route('/api/predict/family-history', methods=['POST'])
def predict_family():
    return predict('family_history')

@app.route('/api/predict/risk-factors', methods=['POST'])
def predict_risk():
    return predict('risk_factors')

@app.route('/api/predict/lifestyle', methods=['POST'])
def predict_lifestyle():
    return predict('lifestyle')

@app.route('/api/predict/allergies', methods=['POST'])
def predict_allergies():
    return predict('allergies')


# ── Add new real patient to knowledge base ───────────────────
@app.route('/api/add-patient', methods=['POST'])
def add_patient():
    """
    Add a real patient to training_patients.csv
    so similarity engine gets smarter over time.

    Request body:
        {
            "patient_id"      : "REAL001",
            "age"             : 45,
            "gender"          : "Male",
            "blood_group"     : "B+",
            "season"          : "Winter",
            "has_history"     : true,
            "medical_problems": "Hypertension | Diabetes",
            "family_history"  : "Father had Diabetes",
            "risk_factors"    : "Smoking | Obesity",
            "lifestyle"       : "Sedentary Lifestyle",
            "allergies"       : "No Known Allergies"
        }
    """
    try:
        import pandas as pd
        import os
        from predictor import MODELS_DIR, df_train, Y, Y_all, \
                              X_train_profile, mlbs, encoders, \
                              scaler, age_group, safe_enc

        data = request.get_json()
        csv_path = f'{MODELS_DIR}/training_patients.csv'

        # Append new patient to CSV
        new_row = pd.DataFrame([{
            'patient_id'      : data.get('patient_id', f'REAL{len(df_train)+1}'),
            'age'             : data.get('age', 30),
            'gender'          : data.get('gender', 'Unknown'),
            'blood_group'     : data.get('blood_group', 'Unknown'),
            'season'          : data.get('season', 'Unknown'),
            'occupation'      : data.get('occupation', 'Unknown'),
            'has_history'     : data.get('has_history', False),
            'medical_problems': data.get('medical_problems', ''),
            'family_history'  : data.get('family_history', ''),
            'risk_factors'    : data.get('risk_factors', ''),
            'lifestyle'       : data.get('lifestyle', ''),
            'allergies'       : data.get('allergies', ''),
        }])

        new_row.to_csv(csv_path, mode='a', header=False, index=False)

        return jsonify({
            'status' : 'success',
            'message': f'Patient added to knowledge base',
            'total_patients': len(df_train) + 1
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Medical ML API — All 5 Tabs")
    print("  Running on http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
