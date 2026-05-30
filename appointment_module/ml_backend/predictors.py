"""
appointment_module/ml_backend/predictors.py
============================================
Django-inline ML prediction functions — the EMR LightGBM engine.

The full implementations live in ``appointment_module/utils.py`` because they
are deeply coupled to Django ORM serializers (SymptomesCatlogSerializer,
MedicineCatlogSerializer, etc.) and database models that live outside
ml_backend.  Moving the implementations here would create circular imports.

This module re-exports the two public prediction functions so that callers can
use either import path:

    # Direct path (implementation lives here conceptually)
    from appointment_module.ml_backend.predictors import predict_emr_fields
    from appointment_module.ml_backend.predictors import predict_drug_usage

    # Or via the legacy path (still works — shims redirect here)
    from appointment_module.utils import predict_emr_fields

What each function does
-----------------------
predict_emr_fields(input_data, top_n, branch, doctor, patient)
    Uses the LightGBM registry (ml_backend/emr_model/model_training/models_registery.py)
    to predict: symptoms, diagnosis, medicines, observation, summary,
    advice, instructions, lab_test.
    Returns DB-resolved serialized objects with confidence, source, reason,
    and co-occurrence alternatives.

predict_drug_usage(drug_name)
    Uses drug-usage Random-Forest models (ml_backend/emr_model/model_training/drug_usage_models.py)
    to predict: intake_id, frequency_id, route_id, type, duration_number, duration_unit.

resolve_drug_usage_objects(prediction)
    Converts predicted IDs from predict_drug_usage() into MedicineSetting
    Django ORM objects via a single DB query.

Do NOT add new logic here — put it in utils.py and re-export it below.
"""

# Re-export from the implementation module.
# Note: appointment_module.utils imports from ml_backend (one-way), so this
# import does NOT create a circular dependency.
from appointment_module.utils import (  # noqa: F401
    predict_emr_fields,
    predict_drug_usage,
    resolve_drug_usage_objects,
)

__all__ = [
    'predict_emr_fields',
    'predict_drug_usage',
    'resolve_drug_usage_objects',
]
