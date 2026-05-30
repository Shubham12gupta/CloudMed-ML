# predictor.py — Medical ML v3 — Hybrid FAISS + RF
# Complaints, Findings, Diagnosis prediction

import pandas as pd
import numpy as np
import joblib
import faiss
import os
from datetime import datetime

# ── Constants ────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'models')
TARGET_TABS  = ['complaints', 'findings', 'diagnosis']
MED_HIST_TABS= ['medical_problems', 'family_history',
                 'risk_factors', 'lifestyle', 'allergies']
VITAL_COLS   = [
    'age', 'temperature', 'pulse', 'spo2',
    'bp_systolic', 'bp_diastolic', 'respiratory_rate',
    'weight_kg', 'height_feet', 'bmi', 'bsa',
    'waist_cm', 'hip_cm', 'waist_hip_ratio', 'head_circumference'
]

# ── Simple cache ─────────────────────────────────────────────
import hashlib, json
_cache     = {}
CACHE_MAX  = 500

def _cache_key(patient, vitals, medical_history, known_tabs, target_tab):
    data = json.dumps({
        'p': patient, 'v': vitals,
        'm': medical_history, 'k': known_tabs,
        't': target_tab
    }, sort_keys=True, default=str)
    return hashlib.md5(data.encode()).hexdigest()

# ── Load models at startup ───────────────────────────────────
print("Loading v3 models...")

models       = {}
mlbs_consult = {}
mlbs_med     = {}

for tab in TARGET_TABS:
    models[tab]       = joblib.load(f'{MODELS_DIR}/{tab}_model.pkl')
    mlbs_consult[tab] = joblib.load(f'{MODELS_DIR}/{tab}_mlb.pkl')
    print(f"  ✅ {tab} model loaded")

for tab in MED_HIST_TABS:
    mlbs_med[tab] = joblib.load(
        f'{MODELS_DIR}/med_history_mlbs/{tab}_mlb.pkl'
    )
print("  ✅ medical history MLBs loaded")

encoders = joblib.load(f'{MODELS_DIR}/profile_encoders.pkl')
scaler   = joblib.load(f'{MODELS_DIR}/scaler.pkl')
print("  ✅ encoders and scaler loaded")

# ── Load FAISS indexes ───────────────────────────────────────
print("Loading FAISS indexes...")
faiss_indexes = {}
for tab in TARGET_TABS:
    faiss_indexes[tab] = faiss.read_index(
        f'{MODELS_DIR}/faiss_{tab}.index'
    )
    print(f"  ✅ faiss_{tab}.index  "
          f"({faiss_indexes[tab].ntotal} vectors)")

# ── Load label matrices ──────────────────────────────────────
print("Loading label matrices...")
Y_consult = {}
for tab in TARGET_TABS:
    Y_consult[tab] = np.load(
        f'{MODELS_DIR}/Y_{tab}.npy'
    ).astype(np.float32)
    print(f"  ✅ Y_{tab}.npy  {Y_consult[tab].shape}")

print("✅ All v3 models loaded — predictor ready\n")

# ── Season auto-detection ────────────────────────────────────
def get_season():
    month = datetime.now().month
    if month in [3,4,5]:    return "Summer"
    elif month in [6,7,8,9]: return "Monsoon"
    elif month in [10,11]:  return "Autumn"
    else:                    return "Winter"

# ── Safe encoder ─────────────────────────────────────────────
def safe_enc(le, val):
    val = str(val)
    return int(le.transform([val])[0]) if val in le.classes_ else 0

# ── City to state mapping ────────────────────────────────────
CITY_STATE = {
    "mumbai": "Maharashtra", "pune": "Maharashtra",
    "nagpur": "Maharashtra", "nashik": "Maharashtra",
    "delhi": "Delhi", "new delhi": "Delhi",
    "bangalore": "Karnataka", "bengaluru": "Karnataka",
    "mysore": "Karnataka",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu",
    "hyderabad": "Telangana", "warangal": "Telangana",
    "kolkata": "West Bengal", "howrah": "West Bengal",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan",
    "ahmedabad": "Gujarat", "surat": "Gujarat",
    "vadodara": "Gujarat",
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh",
    "patna": "Bihar", "gaya": "Bihar",
    "chandigarh": "Punjab", "ludhiana": "Punjab",
    "amritsar": "Punjab",
    "bhubaneswar": "Odisha", "cuttack": "Odisha",
    "guwahati": "Assam", "dibrugarh": "Assam",
    "kochi": "Kerala", "thiruvananthapuram": "Kerala",
    "kozhikode": "Kerala",
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand",
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand",
    "goa": "Goa", "panaji": "Goa",
    "imphal": "Manipur", "shillong": "Meghalaya",
    "vijayawada": "Andhra Pradesh", "visakhapatnam": "Andhra Pradesh",
}

def place_to_state(place_text):
    if not place_text:
        return "Unknown"
    place = place_text.lower().strip()
    for city, state in CITY_STATE.items():
        if city in place:
            return state
    # Check if place itself is a state name
    known_states = list(encoders['state'].classes_)
    for state in known_states:
        if state.lower() in place:
            return state
    return "Unknown"

# ── Build feature vector ─────────────────────────────────────
def build_vector(patient, vitals, medical_history, known_tabs, target_tab):
    """
    Build feature vector for prediction.

    patient: {age, gender, place}
    vitals: {temperature, pulse, spo2, bp_systolic, bp_diastolic,
             respiratory_rate, weight_kg, height_feet, bmi, bsa,
             waist_cm, hip_cm, waist_hip_ratio, head_circumference}
    medical_history: {medical_problems, family_history, risk_factors,
                      lifestyle, allergies}
    known_tabs: {complaints: [...], findings: [...], diagnosis: [...]}
    target_tab: which tab to predict
    """

    # ── Profile features ──────────────────────────────────────
    gender = patient.get('gender', 'Male')
    place  = patient.get('place', '') or patient.get('state', '')
    state  = place_to_state(place)
    season = get_season()   # auto-detected from current date

    g_enc = safe_enc(encoders['gender'], gender)
    s_enc = safe_enc(encoders['state'],  state)
    se_enc= safe_enc(encoders['season'], season)

    cat_vec = np.array([g_enc, s_enc, se_enc], dtype=np.float32)

    # ── Vitals features ───────────────────────────────────────
    def safe_float(val, default):
        try:
            if val is None: return float(default)
            return float(val)
        except:
            return float(default)

    vital_vec = np.array([
        safe_float(vitals.get('age') or patient.get('age'),  30),
        safe_float(vitals.get('temperature'),        98.6),
        safe_float(vitals.get('pulse'),              72),
        safe_float(vitals.get('spo2'),               98),
        safe_float(vitals.get('bp_systolic'),        120),
        safe_float(vitals.get('bp_diastolic'),       80),
        safe_float(vitals.get('respiratory_rate'),   16),
        safe_float(vitals.get('weight_kg'),          65),
        safe_float(vitals.get('height_feet'),        5.5),
        safe_float(vitals.get('bmi'),                22),
        safe_float(vitals.get('bsa'),                1.7),
        safe_float(vitals.get('waist_cm'),           80),
        safe_float(vitals.get('hip_cm'),             95),
        safe_float(vitals.get('waist_hip_ratio'),    0.85),
        safe_float(vitals.get('head_circumference'), 55),
    ], dtype=np.float32)

    # Normalize vitals
    vital_scaled = scaler.transform(vital_vec.reshape(1, -1))[0]

    # ── Medical history features ──────────────────────────────
    med_parts = []
    for tab in MED_HIST_TABS:
        labels = medical_history.get(tab, [])
        if isinstance(labels, str):
            labels = [l.strip() for l in labels.split('|') if l.strip()]
        vec = mlbs_med[tab].transform([labels])[0].astype(np.float32)
        med_parts.append(vec)

    med_vec = np.concatenate(med_parts)

    # ── Cross-functional features ─────────────────────────────
    cross_parts = []
    for tab in TARGET_TABS:
        if tab == target_tab:
            continue
        labels = known_tabs.get(tab, [])
        if isinstance(labels, str):
            labels = [l.strip() for l in labels.split('|') if l.strip()]
        if labels:
            vec = mlbs_consult[tab].transform([labels])[0].astype(np.float32)
        else:
            vec = np.zeros(len(mlbs_consult[tab].classes_), dtype=np.float32)
        cross_parts.append(vec)

    cross_vec = np.concatenate(cross_parts)

    # ── Combine all ───────────────────────────────────────────
    full_vec = np.concatenate([
        cat_vec,        # 3
        vital_scaled,   # 15
        med_vec,        # 112
        cross_vec       # varies per tab
    ]).astype(np.float32)

    return full_vec.reshape(1, -1)


# ── Main prediction function ─────────────────────────────────
def get_recommendations(patient, vitals, medical_history,
                        known_tabs, target_tab, top_n=10):
    """
    Get top N recommendations for target_tab.

    Returns: list of {'label': str, 'confidence': float}
    """

    # Cache check
    key = _cache_key(patient, vitals, medical_history,
                     known_tabs, target_tab)
    if key in _cache:
        return _cache[key]

    # Build feature vector
    X = build_vector(patient, vitals, medical_history,
                     known_tabs, target_tab)

    # ── FAISS search ──────────────────────────────────────────
    X_norm = X.copy()
    faiss.normalize_L2(X_norm)

    D, I      = faiss_indexes[target_tab].search(X_norm, k=21)
    top20_idx = I[0][1:]   # skip self (position 0)

    # ── Frequency from similar patients ───────────────────────
    freq = Y_consult[target_tab][top20_idx].mean(axis=0)

    # ── RF confidence ─────────────────────────────────────────
    rf_probas = models[target_tab].predict_proba(X)
    rf_scores = np.array([
        float(p[0][1]) if p.shape[1] > 1 else 0.0
        for p in rf_probas
    ], dtype=np.float32)

    # ── Hybrid score ──────────────────────────────────────────
    final  = (0.6 * freq) + (0.4 * rf_scores)
    labels = mlbs_consult[target_tab].classes_

    # Filter already selected
    already = set(known_tabs.get(target_tab, []))

    results = [
        {'label': label, 'confidence': round(float(final[i]) * 100, 1)}
        for i, label in enumerate(labels)
        if label not in already
    ]

    results = sorted(results, key=lambda x: -x['confidence'])[:top_n]

    # Cache result
    if len(_cache) >= CACHE_MAX:
        for k in list(_cache.keys())[:100]:
            del _cache[k]
    _cache[key] = results

    return results