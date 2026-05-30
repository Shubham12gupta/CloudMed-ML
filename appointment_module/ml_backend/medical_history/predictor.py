# predictor.py — Hybrid Recommendation Engine (All 5 Tabs) — Optimized

import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import json

# ── Constants ────────────────────────────────────────────────
TABS = [
    'medical_problems',
    'family_history',
    'risk_factors',
    'lifestyle',
    'allergies'
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Simple in-memory cache ───────────────────────────────────
_cache = {}
CACHE_MAX = 500  # max cached results

def _cache_key(patient_profile, known_tabs, target_tab):
    data = json.dumps({
        'p' : patient_profile,
        'k' : known_tabs,
        't' : target_tab
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()

# ── Load all models at startup ───────────────────────────────
print("Loading models...")

models = {}
mlbs   = {}

for tab in TABS:
    models[tab] = joblib.load(f'{MODELS_DIR}/{tab}_model.pkl')
    mlbs[tab]   = joblib.load(f'{MODELS_DIR}/{tab}_mlb.pkl')
    print(f"  ✅ {tab} model loaded")

encoders = joblib.load(f'{MODELS_DIR}/profile_encoders.pkl')
scaler   = joblib.load(f'{MODELS_DIR}/scaler.pkl')
print("  ✅ encoders and scaler loaded")

# ── Load training patients ───────────────────────────────────
print("Loading training patients...")
df_train = pd.read_csv(f'{MODELS_DIR}/training_patients.csv')
df_train[df_train.select_dtypes(include='object').columns] = \
    df_train.select_dtypes(include='object').fillna('')
print(f"  ✅ {len(df_train)} training patients loaded")

# ── Build label matrices ─────────────────────────────────────
print("Building label matrices...")
Y = {}
for tab in TABS:
    df_train[tab] = df_train[tab].fillna('')
    tab_lists     = df_train[tab].apply(
        lambda x: [i.strip() for i in x.split(' | ') if i.strip()]
    )
    Y[tab] = mlbs[tab].transform(tab_lists).astype(np.float32)

Y_all = np.hstack([Y[tab] for tab in TABS]).astype(np.float32)
print(f"  ✅ Label matrices ready — Y_all shape: {Y_all.shape}")

# ── Build normalized profile matrix ─────────────────────────
print("Building profile matrix...")

PROFILE_COLS = ['age', 'age_group', 'has_history_int',
                'gender_enc', 'blood_group_enc', 'season_enc']

def age_group_fn(age):
    if age <= 18:   return 0
    elif age <= 30: return 1
    elif age <= 45: return 2
    elif age <= 60: return 3
    else:           return 4

def safe_enc(le, val):
    val = str(val)
    return int(le.transform([val])[0]) if val in le.classes_ else 0

# Build profile rows
rows = []
for _, row in df_train.iterrows():
    rows.append([
        float(row['age']),
        float(age_group_fn(int(row['age']))),
        float(bool(row['has_history'])),
        float(safe_enc(encoders['gender'],      row['gender'])),
        float(safe_enc(encoders['blood_group'], row['blood_group'])),
        float(safe_enc(encoders['season'],      row['season'])),
    ])

X_raw            = np.array(rows, dtype=np.float32)
X_train_scaled   = scaler.transform(X_raw).astype(np.float32)

# Pre-normalize for fast cosine similarity via dot product
norms            = np.linalg.norm(X_train_scaled, axis=1, keepdims=True)
norms            = np.where(norms == 0, 1e-9, norms)
X_train_normed   = (X_train_scaled / norms).astype(np.float32)

# Pre-compute row sums of Y_all for Jaccard
Y_all_rowsums    = Y_all.sum(axis=1).astype(np.float32)

print(f"  ✅ Profile matrix ready — shape: {X_train_normed.shape}")
print("✅ All models loaded — predictor ready\n")


# ── Helper: encode new patient profile ──────────────────────
def encode_patient(patient_profile):
    age  = int(patient_profile['age'])
    ag   = age_group_fn(age)
    hist = int(bool(patient_profile.get('has_history', False)))
    g    = safe_enc(encoders['gender'],      patient_profile.get('gender',      'Male'))
    b    = safe_enc(encoders['blood_group'], patient_profile.get('blood_group', 'O+'))
    s    = safe_enc(encoders['season'],      patient_profile.get('season',      'Winter'))

    raw    = np.array([[age, ag, hist, g, b, s]], dtype=np.float32)
    scaled = scaler.transform(raw).astype(np.float32)
    norm   = np.linalg.norm(scaled)
    normed = scaled / (norm if norm > 0 else 1e-9)
    return raw, normed


# ── Main recommendation function ─────────────────────────────
def get_recommendations(patient_profile, known_tabs, target_tab, top_n=10):
    """
    Get top N recommendations for target_tab.

    Args:
        patient_profile : dict — age, gender, blood_group, season, has_history
        known_tabs      : dict — already filled tabs
                          e.g. {'medical_problems': ['Hypertension', 'Diabetes']}
        target_tab      : str  — which tab to predict for
        top_n           : int  — number of suggestions (default 10)

    Returns:
        list of (label, confidence%) sorted by confidence descending
    """

    # ── Cache check ──────────────────────────────────────────
    cache_key = _cache_key(patient_profile, known_tabs, target_tab)
    if cache_key in _cache:
        return _cache[cache_key]

    # ── Step 1: Encode patient profile ───────────────────────
    profile_raw, profile_normed = encode_patient(patient_profile)

    # ── Step 2: Fast cosine similarity (dot product) ─────────
    c_scores = (X_train_normed @ profile_normed.T).flatten()

    # ── Step 3: Jaccard similarity on known labels ───────────
    known_vec = np.zeros(Y_all.shape[1], dtype=np.float32)
    col_offset = 0
    for tab in TABS:
        n_labels = Y[tab].shape[1]
        if tab in known_tabs and known_tabs[tab]:
            for label in known_tabs[tab]:
                if label in mlbs[tab].classes_:
                    idx = list(mlbs[tab].classes_).index(label)
                    known_vec[col_offset + idx] = 1.0
        col_offset += n_labels

    known_sum = float(known_vec.sum())

    if known_sum > 0:
        # Fast matrix-vector multiply
        intersection = (Y_all @ known_vec).astype(np.float32)
        union        = Y_all_rowsums + known_sum - intersection
        union        = np.where(union == 0, 1e-9, union)
        j_scores     = (intersection / union).astype(np.float32)
    else:
        j_scores = np.zeros(len(df_train), dtype=np.float32)

    # ── Step 4: Dynamic weighting ────────────────────────────
    tabs_filled = sum(1 for t in TABS if t in known_tabs and known_tabs[t])
    cosine_w    = max(0.1, 0.4 - (tabs_filled * 0.06))
    jaccard_w   = 1.0 - cosine_w

    hybrid = (cosine_w * c_scores) + (jaccard_w * j_scores)

    # ── Step 5: Top 20 similar patients (faster than 50) ─────
    top_k      = 20
    top_k_idx  = np.argpartition(hybrid, -top_k)[-top_k:]
    top_k_idx  = top_k_idx[np.argsort(hybrid[top_k_idx])[::-1]]

    # ── Step 6: Frequency score from similar patients ────────
    freq_scores = Y[target_tab][top_k_idx].mean(axis=0)

    # ── Step 7: RF confidence scores ─────────────────────────
    rf_probas = models[target_tab].predict_proba(profile_raw)
    rf_scores = np.array([
        float(p[0][1]) if p.shape[1] > 1 else 0.0
        for p in rf_probas
    ], dtype=np.float32)

    # ── Step 8: Final combined score ─────────────────────────
    final_scores = (0.6 * freq_scores) + (0.4 * rf_scores)

    # ── Step 9: Filter already selected + return top N ───────
    already = set(known_tabs.get(target_tab, []))
    labels  = mlbs[target_tab].classes_

    results = [
        (label, round(float(final_scores[i]) * 100, 1))
        for i, label in enumerate(labels)
        if label not in already
    ]

    results = sorted(results, key=lambda x: -x[1])[:top_n]

    # ── Cache result ─────────────────────────────────────────
    if len(_cache) >= CACHE_MAX:
        # Remove oldest 100 entries
        keys = list(_cache.keys())[:100]
        for k in keys:
            del _cache[k]

    _cache[cache_key] = results
    return results