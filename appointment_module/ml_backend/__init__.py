"""
appointment_module/ml_backend/
==============================
All Medical-ML integration code lives under this package.

Sub-modules
-----------
http_client.py
    HTTP client for Flask microservices:
      • get_suggestions()              — ML v2 (medical history, port 5000)
      • get_consultation_suggestions() — ML v3 (consultation, port 5001)
      • add_to_ml()                    — push patient data to ML v2 knowledge base
      • add_consultation_to_ml()       — push consultation data to ML v3 knowledge base
      • VALID_TABS / VALID_CONSULTATION_TABS

predictors.py
    Django-inline ML prediction (LightGBM EMR engine):
      • predict_emr_fields()           — predict symptoms, diagnosis, medicines, etc.
      • predict_drug_usage()           — predict intake, frequency, route, duration
      • resolve_drug_usage_objects()   — convert predicted IDs to Django ORM objects

view_helpers.py
    Pure helper functions for PatientMLInsightsView (in views.py):
      • _safe_float()
      • _build_vitals_payload()
      • _build_medical_history_from_prescriptions()

emr_model/
    LightGBM model registry and drug-usage Random-Forest models.
    Key entry point: emr_model/model_training/models_registery.py → registry

medical_history/
    Flask app for ML v2 (medical history suggestions, port 5000).

consultation/
    Flask app for ML v3 (consultation suggestions — FAISS + Random Forest, port 5001).

shared/config.py
    Shared configuration: ML_V2_PORT, ML_V3_PORT, S3 paths.
"""
