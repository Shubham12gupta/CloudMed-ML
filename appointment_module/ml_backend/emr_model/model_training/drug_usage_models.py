import joblib
from functools import lru_cache
import os

BASE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../models'
)

@lru_cache(maxsize=1)
def get_drug_usage_assets():
    """
    Load and cache all models, vectorizers, and label encoders for drug prediction.
    """
    
    return {
        "usage_model": joblib.load(os.path.join(BASE_PATH, "lgb_drug_usage_model.pkl")),
        "route_model": joblib.load(os.path.join(BASE_PATH, "lgb_route_model.pkl")),
        "usage_tfidf": joblib.load(os.path.join(BASE_PATH, "drug_name_tfidf.pkl")),
        "intake_encoder": joblib.load(os.path.join(BASE_PATH, "intake_encoder.pkl")),
        "frequency_encoder": joblib.load(os.path.join(BASE_PATH, "frequency_encoder.pkl")),
        "route_encoder": joblib.load(os.path.join(BASE_PATH, "route_encoder.pkl")),

        "type_model": joblib.load(os.path.join(BASE_PATH, "lgb_drug_type_model.pkl")),
        "type_tfidf": joblib.load(os.path.join(BASE_PATH, "type_tfidf.pkl")),
        "type_encoder": joblib.load(os.path.join(BASE_PATH, "type_label_encoder.pkl")),

        "duration_number_model": joblib.load(os.path.join(BASE_PATH, "lgb_duration_model.pkl")),
        "duration_number_tfidf": joblib.load(os.path.join(BASE_PATH, "duration_tfidf.pkl")),

        "duration_unit_model": joblib.load(os.path.join(BASE_PATH, "lgb_duration_unit_model.pkl")),
        "duration_unit_tfidf": joblib.load(os.path.join(BASE_PATH, "duration_unit_tfidf.pkl")),
        "duration_unit_encoder": joblib.load(os.path.join(BASE_PATH, "duration_unit_label_encoder.pkl")),
    }
