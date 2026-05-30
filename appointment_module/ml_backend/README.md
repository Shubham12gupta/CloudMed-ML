# ML Backend Services

Two independent Flask microservices that provide AI-powered suggestions
to the main Django backend. They are completely isolated from Django —
they only communicate via HTTP. -- yes

---

## Structure

```
ml_backend/                         ← SINGLE source of truth for ALL ML code
│
├── emr_model/                      ← Django-inline: EMR prescription model
│   ├── model_training/             Python modules imported directly by Django
│   │   ├── models_registery.py
│   │   └── drug_usage_models.py
│   ├── models/                     GITIGNORED — .pkl files loaded by Django
│   └── .gitignore
│
├── equeue_model/                   ← Django-inline: eQueue wait-time model
│   ├── model_training/
│   ├── models/                     GITIGNORED — .pkl file loaded by Django
│   └── .gitignore
│
├── medical_history/                ← Flask ML v2  (port 5000)
│   ├── app.py                      Flask server — run this
│   ├── predictor.py                ML logic (RF + patient similarity)
│   ├── requirements.txt
│   ├── models/                     GITIGNORED — populated from S3
│   └── .gitignore
│
├── consultation/                   ← Flask ML v3  (port 5001)
│   ├── app.py                      Flask server — run this
│   ├── predictor.py                ML logic (FAISS + RF hybrid)
│   ├── requirements.txt
│   ├── models/                     GITIGNORED — populated from S3
│   └── .gitignore
│
└── shared/
    ├── config.py                   Central port / S3 / env config
    ├── ec2_setup.sh                One-time EC2 setup (venv + S3 sync)
    ├── create_systemd_services.sh  Create + start systemd services
    └── .gitignore
```

---

## Services at a Glance

| Folder | Type | Port | Purpose |
|---|---|---|---|
| `emr_model/` | Django-inline | — | EMR prescription suggestions (loaded directly in Django process) |
| `equeue_model/` | Django-inline | — | eQueue wait-time prediction (loaded directly in Django process) |
| `medical_history/` | Flask service | 5000 | Medical history tab suggestions (5 tabs) |
| `consultation/` | Flask service | 5001 | Complaints, findings, diagnosis suggestions |

**Django-inline** = Python module imported by `appointment_module/utils.py` directly.
**Flask service** = Separate process; Django calls it via HTTP through `ml_utils.py`.

---

## API Endpoints (both services)

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check — confirm service is up |
| `/api/predict/<tab>` | POST | Get AI suggestions for a tab |
| `/api/add-patient` | POST | (v2) Feed real patient data for learning |
| `/api/add-consultation` | POST | (v3) Feed real consultation for learning |

---

## Quick Start (local dev)

```bash
# ML v2
cd medical_history
pip install -r requirements.txt
# Place models/ from S3 here first
python app.py          # starts on port 5000

# ML v3
cd consultation
pip install -r requirements.txt
# Place models/ from S3 here first
python app.py          # starts on port 5001
```

---

## EC2 Deployment (one-time)

```bash
# 1. Setup both services (venv + S3 model download)
export MODELS_S3_BUCKET=your-bucket-name
bash shared/ec2_setup.sh

# 2. Create systemd services (auto-start on reboot)
sudo bash shared/create_systemd_services.sh

# 3. Verify both are running
curl http://localhost:5000/health
curl http://localhost:5001/health
```

---

## Model Files (NOT in git)

All `.pkl`, `.csv`, `.npy`, `.index` files live in S3.
Never commit them to git — they are 200MB–700MB each.

```
S3 layout (example):
  s3://your-bucket/ml-models/v2/latest/   ← v2 models
  s3://your-bucket/ml-models/v3/latest/   ← v3 models
```

---

## Django Integration

Django calls these services from:
- `appointment_module/ml_utils.py`

Settings keys (in `.env.json`):
```json
"ml_v2_base": "http://localhost:5000/api",
"ml_v3_base": "http://localhost:5001/api",
"ml_testing_mode": false
```

Set `ml_testing_mode: false` in production to enable live learning.

---

## Firewall Rules

```
Open ports  : 22, 80, 443, 8000
Block ports : 5000, 5001  (internal EC2 traffic only)
```
