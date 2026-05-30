"""
shared/config.py
================
Central configuration for all ML backend services.

Services in ml_backend/:
  ┌─────────────────┬──────────────────────────────────────────┬───────────────┐
  │ Folder          │ Purpose                                  │ Type          │
  ├─────────────────┼──────────────────────────────────────────┼───────────────┤
  │ emr_model/      │ Prescription suggestions (EMR)           │ Django-inline │
  │ equeue_model/   │ Wait-time prediction (eQueue)            │ Django-inline │
  │ medical_history/│ Medical history tab suggestions (v2)     │ Flask :5000   │
  │ consultation/   │ Complaints/Findings/Diagnosis (v3)       │ Flask :5001   │
  └─────────────────┴──────────────────────────────────────────┴───────────────┘

Override at runtime via environment variables:
    ML_V2_PORT=5000 ML_V3_PORT=5001 python app.py
"""
import os

# ── Flask service ports ────────────────────────────────────────────────────
ML_V2_PORT = int(os.environ.get('ML_V2_PORT', 5000))   # medical_history service
ML_V3_PORT = int(os.environ.get('ML_V3_PORT', 5001))   # consultation service

# ── S3 Bucket for model files ──────────────────────────────────────────────
# Set via environment variable on EC2 before starting services.
# Example: export MODELS_S3_BUCKET=your-bucket-name
MODELS_S3_BUCKET        = os.environ.get('MODELS_S3_BUCKET', '')
MODELS_S3_PREFIX_V2     = os.environ.get('MODELS_S3_PREFIX_V2',  'ml-models/v2/latest/')
MODELS_S3_PREFIX_V3     = os.environ.get('MODELS_S3_PREFIX_V3',  'ml-models/v3/latest/')
MODELS_S3_PREFIX_EMR    = os.environ.get('MODELS_S3_PREFIX_EMR', 'ml-models/emr/latest/')
MODELS_S3_PREFIX_EQUEUE = os.environ.get('MODELS_S3_PREFIX_EQUEUE', 'ml-models/equeue/latest/')

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# ── CORS origins (for Flask) ───────────────────────────────────────────────
# In production these services should only accept calls from the Django backend
# (internal EC2 traffic). Set ALLOWED_ORIGINS to '*' only during local dev.
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
