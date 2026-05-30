# app.py — Medical ML API v3
# Complaints, Findings, Diagnosis — Port 5001
# Lets See now

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import get_recommendations, TARGET_TABS
import traceback

app = Flask(__name__)
CORS(app)

# ── Health check ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'healthy',
        'version': 'v3',
        'tabs'   : TARGET_TABS,
        'message': 'Medical ML v3 API — Complaints, Findings, Diagnosis ready'
    })


# ── Generic predict endpoint ─────────────────────────────────
@app.route('/api/predict/<tab_name>', methods=['POST'])
def predict(tab_name):
    """
    POST /api/predict/complaints
    POST /api/predict/findings
    POST /api/predict/diagnosis

    Request body:
    {
        "patient": {
            "age"   : 24,
            "gender": "Male",
            "place" : "Mumbai"
        },
        "vitals": {
            "temperature"      : 105,
            "pulse"            : 105,
            "spo2"             : 98,
            "bp_systolic"      : 120,
            "bp_diastolic"     : 80,
            "respiratory_rate" : 18,
            "weight_kg"        : 70,
            "height_feet"      : 5.5,
            "bmi"              : 24.5,
            "bsa"              : 1.75,
            "waist_cm"         : 85,
            "hip_cm"           : 95,
            "waist_hip_ratio"  : 0.89,
            "head_circumference": 55
        },
        "medical_history": {
            "medical_problems": ["Diabetes", "Hypertension"],
            "family_history"  : ["Father had Diabetes"],
            "risk_factors"    : ["Smoking"],
            "lifestyle"       : ["Sedentary Lifestyle"],
            "allergies"       : ["No Known Allergies"]
        },
        "known_tabs": {
            "complaints": ["Fever", "Headache"],
            "findings"  : [],
            "diagnosis" : []
        }
    }

    Response:
    {
        "tab": "findings",
        "suggestions": [
            {"label": "High Temperature", "confidence": 72.3},
            {"label": "High Pulse",       "confidence": 65.1},
            ...
        ]
    }
    """

    if tab_name not in TARGET_TABS:
        return jsonify({
            'error': f'Invalid tab: {tab_name}. Must be one of {TARGET_TABS}'
        }), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body sent'}), 400

        patient         = data.get('patient', {})
        vitals          = data.get('vitals', {})
        medical_history = data.get('medical_history', {})
        known_tabs      = data.get('known_tabs', {})

        # age can be in patient or vitals
        if 'age' not in vitals and 'age' in patient:
            vitals['age'] = patient['age']

        # Validate minimum required
        if not patient.get('age') and not vitals.get('age'):
            return jsonify({
                'error'  : 'age is required',
                'example': {'patient': {'age': 24, 'gender': 'Male'}}
            }), 400

        results = get_recommendations(
            patient         = patient,
            vitals          = vitals,
            medical_history = medical_history,
            known_tabs      = known_tabs,
            target_tab      = tab_name,
            top_n           = 10
        )

        return jsonify({
            'tab'        : tab_name,
            'suggestions': results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Individual endpoints ──────────────────────────────────────
@app.route('/api/predict/complaints', methods=['POST'])
def predict_complaints():
    return predict('complaints')

@app.route('/api/predict/findings', methods=['POST'])
def predict_findings():
    return predict('findings')

@app.route('/api/predict/diagnosis', methods=['POST'])
def predict_diagnosis():
    return predict('diagnosis')


# ── Add real consultation for live learning ───────────────────
@app.route('/api/add-consultation', methods=['POST'])
def add_consultation():
    """
    Call this after doctor saves a consultation.
    Appends real patient data to training CSV.
    System learns from real consultations automatically.

    Request body:
    {
        "consultation_id": "C001",
        "age"            : 24,
        "gender"         : "Male",
        "place"          : "Mumbai",
        "temperature"    : 105,
        "pulse"          : 105,
        "spo2"           : 98,
        "bp_systolic"    : 120,
        "bp_diastolic"   : 80,
        "respiratory_rate": 18,
        "weight_kg"      : 70,
        "height_feet"    : 5.5,
        "bmi"            : 24.5,
        "bsa"            : 1.75,
        "waist_cm"       : 85,
        "hip_cm"         : 95,
        "waist_hip_ratio": 0.89,
        "head_circumference": 55,
        "medical_problems": "Diabetes | Hypertension",
        "family_history" : "Father had Diabetes",
        "risk_factors"   : "Smoking",
        "lifestyle"      : "Sedentary Lifestyle",
        "allergies"      : "No Known Allergies",
        "complaints"     : "Fever | Headache | Body Ache",
        "findings"       : "High Temperature | High Pulse",
        "diagnosis"      : "Malaria"
    }
    """
    try:
        import pandas as pd
        from predictor import MODELS_DIR

        data     = request.get_json()
        csv_path = f'{MODELS_DIR}/training_consultations.csv'

        new_row = pd.DataFrame([{
            'record_id'         : data.get('consultation_id', 'REAL001'),
            'age'               : data.get('age', 30),
            'gender'            : data.get('gender', 'Unknown'),
            'state'             : data.get('place', 'Unknown'),
            'season'            : data.get('season', 'Unknown'),
            'temperature'       : data.get('temperature', 98.6),
            'pulse'             : data.get('pulse', 72),
            'spo2'              : data.get('spo2', 98),
            'bp_systolic'       : data.get('bp_systolic', 120),
            'bp_diastolic'      : data.get('bp_diastolic', 80),
            'respiratory_rate'  : data.get('respiratory_rate', 16),
            'weight_kg'         : data.get('weight_kg', 65),
            'height_feet'       : data.get('height_feet', 5.5),
            'bmi'               : data.get('bmi', 22),
            'bsa'               : data.get('bsa', 1.7),
            'waist_cm'          : data.get('waist_cm', 80),
            'hip_cm'            : data.get('hip_cm', 95),
            'waist_hip_ratio'   : data.get('waist_hip_ratio', 0.85),
            'head_circumference': data.get('head_circumference', 55),
            'medical_problems'  : data.get('medical_problems', ''),
            'family_history'    : data.get('family_history', ''),
            'risk_factors'      : data.get('risk_factors', ''),
            'lifestyle'         : data.get('lifestyle', ''),
            'allergies'         : data.get('allergies', ''),
            'complaints'        : data.get('complaints', ''),
            'findings'          : data.get('findings', ''),
            'diagnosis'         : data.get('diagnosis', ''),
        }])

        new_row.to_csv(csv_path, mode='a', header=False, index=False)

        return jsonify({
            'status' : 'success',
            'message': 'Consultation added to knowledge base'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Medical ML API v3")
    print("  Complaints + Findings + Diagnosis")
    print("  Running on http://localhost:5001")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
