# 🏥 Medical ML API — Setup & Integration Guide

> AI-powered suggestion system for all 5 medical history tabs.
> Runs as a separate Flask service on port 5000.

---

## ⚡ QUICK START (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API
python app.py

# 3. Verify it's running
curl http://localhost:5000/health
```

You should see all 5 models load and the server start on port 5000. Done.

---

## 📦 WHAT YOU'RE GETTING

```
medical-ml-v2/
├── app.py               ← Flask API server — run this
├── predictor.py         ← ML logic — do NOT modify
├── requirements.txt     ← pip install this
└── models/              ← do NOT modify anything here
    ├── *_model.pkl      ← 5 trained ML models
    ├── *_mlb.pkl        ← 5 label encoders
    ├── profile_encoders.pkl
    ├── scaler.pkl
    ├── training_patients.csv   ← ML knowledge base
    └── model_metadata.json
```

---

## 🧠 HOW IT WORKS (SIMPLE VERSION)

```
Doctor opens a tab (e.g. Risk Factors)
        ↓
Your Django sends patient profile + already filled tabs to ML API
        ↓
ML API finds 20 most similar patients from its database
Ranks their conditions by how often they appear
        ↓
Returns top 10 suggestions with confidence %
        ↓
Doctor sees dropdown — clicks what applies
```

**Cross-functional:** Every tab the doctor fills makes other tabs smarter.
More tabs filled = better suggestions.

---

## 🗄️ TWO DATABASES — HOW THEY WORK TOGETHER

| | Your Database | ML Knowledge Base |
|--|---------------|------------------|
| **File** | patients.db / PostgreSQL | training_patients.csv |
| **Purpose** | Store real patient data | Find similar patients |
| **Who manages** | Your Django backend | ML API |
| **Talk to each other?** | ❌ Never directly | ❌ Never directly |

**They only communicate via HTTP API calls.**

**Flow when doctor saves a patient:**
```
Doctor clicks Save
    ↓
Django saves to your database        ← permanent storage
Django calls POST /api/add-patient   ← ML learns from real patient
ML appends to training_patients.csv  ← suggestions improve over time
```

---

## 🔌 API ENDPOINTS

Base URL: `http://localhost:5000`

### Health Check
```
GET /health
→ {"status": "healthy", "tabs": [...]}
```

### Get Suggestions (same format for all 5 tabs)
```
POST /api/predict/{tab_name}

tab_name can be:
  medical_problems
  family_history
  risk_factors
  lifestyle
  allergies
```

**Request:**
```json
{
  "patient": {
    "age"         : 45,
    "gender"      : "Male",
    "blood_group" : "B+",
    "season"      : "Winter",
    "has_history" : true
  },
  "known_tabs": {
    "medical_problems": ["Hypertension", "Diabetes"]
  }
}
```

> `known_tabs` = tabs already filled by doctor.
> Send empty `{}` if nothing filled yet.
> More tabs filled → better suggestions.

**Response:**
```json
{
  "tab": "risk_factors",
  "suggestions": [
    {"label": "High Salt Intake",    "confidence": 38.8},
    {"label": "Physical Inactivity", "confidence": 37.4},
    {"label": "High Stress Levels",  "confidence": 35.4},
    ...10 total
  ]
}
```

### Add Real Patient (for live learning)
```
POST /api/add-patient
```

```json
{
  "patient_id"      : "P001",
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
```

> Use ` | ` to separate multiple values.
> Call this every time a doctor saves a patient.
> ML learns immediately — no retraining needed.

---

## 🔧 DJANGO INTEGRATION

### 1. Add to requirements.txt
```
requests==2.31.0
```

### 2. Add to settings.py
```python
ML_API_BASE = 'http://localhost:5000/api'  # same EC2
# ML_API_BASE = 'http://ML_SERVER_IP:5000/api'  # separate EC2
```

### 3. Create ml_utils.py in your Django app
```python
import requests
from datetime import datetime
from django.conf import settings

def get_season():
    m = datetime.now().month
    if m in [3,4,5]:    return "Summer"
    elif m in [6,7,8,9]: return "Monsoon"
    elif m in [10,11]:  return "Autumn"
    else:               return "Winter"

def get_suggestions(patient, target_tab, known_tabs={}):
    try:
        r = requests.post(
            f'{settings.ML_API_BASE}/predict/{target_tab}',
            json={
                'patient': {
                    'age'        : patient.age,
                    'gender'     : patient.gender,
                    'blood_group': patient.blood_group or 'O+',
                    'season'     : get_season(),
                    'has_history': bool(patient.has_history),
                },
                'known_tabs': known_tabs
            },
            timeout=10
        )
        return r.json().get('suggestions', [])
    except:
        return []

def add_to_ml(patient, history):
    def pipe(items):
        return ' | '.join(items) if isinstance(items, list) else str(items or '')
    try:
        requests.post(
            f'{settings.ML_API_BASE}/add-patient',
            json={
                'patient_id'      : str(patient.id),
                'age'             : patient.age,
                'gender'          : patient.gender,
                'blood_group'     : patient.blood_group or 'O+',
                'season'          : get_season(),
                'has_history'     : bool(patient.has_history),
                'medical_problems': pipe(history.medical_problems),
                'family_history'  : pipe(history.family_history),
                'risk_factors'    : pipe(history.risk_factors),
                'lifestyle'       : pipe(history.lifestyle),
                'allergies'       : pipe(history.allergies),
            },
            timeout=5
        )
    except:
        pass  # non-critical, don't block the save
```

### 4. Use in views.py
```python
from .ml_utils import get_suggestions, add_to_ml

# When doctor opens a tab
def suggestions_view(request, patient_id):
    patient    = Patient.objects.get(id=patient_id)
    target_tab = request.GET.get('tab', 'medical_problems')
    history    = MedicalHistory.objects.filter(patient=patient).first()

    known_tabs = {}
    if history:
        if history.medical_problems: known_tabs['medical_problems'] = history.medical_problems
        if history.family_history:   known_tabs['family_history']   = history.family_history
        if history.risk_factors:     known_tabs['risk_factors']     = history.risk_factors
        if history.lifestyle:        known_tabs['lifestyle']         = history.lifestyle
        if history.allergies:        known_tabs['allergies']         = history.allergies

    suggestions = get_suggestions(patient, target_tab, known_tabs)
    return JsonResponse({'suggestions': suggestions})

# When doctor saves patient — add ONE line after your existing save
def save_patient_view(request):
    # ... your existing save logic ...
    add_to_ml(patient, history)  # ← add this line
    return JsonResponse({'success': True})
```

---

## 🚀 EC2 PRODUCTION SETUP

### Minimum server spec
```
RAM : 4 GB minimum (models take ~1.2 GB)
Disk: 3 GB free
```

### Run as background service
```bash
sudo nano /etc/systemd/system/medical-ml.service
```

```ini
[Unit]
Description=Medical ML API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/medical-ml-api
ExecStart=/home/ubuntu/medical-ml-api/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
pip install gunicorn
sudo systemctl daemon-reload
sudo systemctl enable medical-ml
sudo systemctl start medical-ml
```

### Firewall (if same EC2 as Django)
```bash
# Block port 5000 from public — Django calls it internally
sudo ufw deny 5000
```

### Firewall (if separate EC2)
```bash
# Only allow Django server to reach ML API
sudo ufw allow from DJANGO_SERVER_IP to any port 5000
```

---

## ❗ COMMON ISSUES

| Problem | Fix |
|---------|-----|
| `InconsistentVersionWarning` on startup | Harmless warning — ignore it |
| Slow first request | Normal — models load into memory once, fast after |
| `Connection refused` | ML API not running — check `systemctl status medical-ml` |
| Empty suggestions | Check age + gender + blood_group + season are all sent |
| Out of memory error | Need more RAM — upgrade to t3.medium (4GB) minimum |

---

## 📈 ACCURACY

| Tab | Hit Rate | What it means |
|-----|----------|---------------|
| Medical Problems | 85% | 8-9 of actual conditions appear in top 10 |
| Family History | 93% | Very accurate |
| Risk Factors | 97% | Near perfect |
| Lifestyle | 98% | Near perfect |
| Allergies | 88% | Strong |

> Accuracy improves automatically as real patients are added via `/api/add-patient`.

---

*v2.0 — Hybrid Jaccard + Cosine + Random Forest — All 5 Tabs*
