import os
import joblib
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

BASE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../models'
)


class ModelRegistry:
    def __init__(self):
        try:
            # MODELS + MLBS
            self.symptoms_model = self._load_boosters('lgb_symptoms_model.pkl')
            self.symptoms_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_symptoms.pkl'))

            self.diagnosis_model = self._load_boosters('lgb_diagnosis_model.pkl')
            self.diagnosis_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_diagnosis.pkl'))

            self.medicine_model = self._load_boosters('lgb_medicines_model.pkl')
            self.medicine_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_medicines.pkl'))

            self.observation_model = self._load_boosters('lgb_observation_model.pkl')
            self.observation_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_observation.pkl'))

            self.summary_model = self._load_boosters('lgb_summary_model.pkl')
            self.summary_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_summary.pkl'))

            self.advice_model = self._load_boosters('lgb_advice_model.pkl')
            self.advice_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_advice.pkl'))

            self.instructions_model = self._load_boosters('lgb_instructions_model.pkl')
            self.instructions_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_instructions.pkl'))

            self.lab_test_model = self._load_boosters('lgb_lab_test_model.pkl')
            self.lab_test_mlb = joblib.load(os.path.join(BASE_PATH, 'mlb_lab_test.pkl'))

            # ENCODERS + SCALER
            self.encoders = {
                "patient_gender": joblib.load(os.path.join(BASE_PATH, "label_encoder_patient_gender.pkl")),
                "patient_city": joblib.load(os.path.join(BASE_PATH, "label_encoder_patient_city.pkl")),
                "patient_blood_group": joblib.load(os.path.join(BASE_PATH, "label_encoder_patient_blood_group.pkl")),
                "doctor_practice_specialty": joblib.load(os.path.join(BASE_PATH, "label_encoder_doctor_practice_specialty.pkl")),
                "doctor_state": joblib.load(os.path.join(BASE_PATH, "label_encoder_doctor_state.pkl")),
                "doctor_country": joblib.load(os.path.join(BASE_PATH, "label_encoder_doctor_country.pkl")),
            }
            self.scaler = joblib.load(os.path.join(BASE_PATH, "feature_scaler.pkl"))
        except Exception as e:
            print(f"Error loading ML models: {e}")
            # Set to None or empty
            self.symptoms_model = None
            self.symptoms_mlb = None
            self.diagnosis_model = None
            self.diagnosis_mlb = None
            self.medicine_model = None
            self.medicine_mlb = None
            self.observation_model = None
            self.observation_mlb = None
            self.summary_model = None
            self.summary_mlb = None
            self.advice_model = None
            self.advice_mlb = None
            self.instructions_model = None
            self.instructions_mlb = None
            self.lab_test_model = None
            self.lab_test_mlb = None
            self.encoders = {}
            self.scaler = None

    def _load_boosters(self, filename):
        """
        Load a list of LightGBM Booster objects from disk.
        """
        if lgb is None:
            return None
        boosters = joblib.load(os.path.join(BASE_PATH, filename))
        if isinstance(boosters, lgb.Booster):
            # wrap single booster in a list for backward compatibility
            boosters = [boosters]
        return boosters


# SINGLETON (created ONCE per Django process)
registry = ModelRegistry()
# registry = None   #Temporarily disabled due to model loading issues
