import os
import joblib
import numpy as np

class CombinedEMRModel:
    """
    Combined multi-label model for EMR fields.
    Returns top N predictions per field
    """

    def __init__(self, models_dict, mlbs_dict):
        self.models = models_dict
        self.mlbs = mlbs_dict

    def predict(self, X, top_n=15):
        """
        Predict top N labels for each EMR field.

        Args:
            X: preprocessed input features (DataFrame)
            top_n: number of labels to return per field

        Returns:
            dict: {field_name: list of top N predictions per sample}
        """
        results = {}

        for field, model in self.models.items():
            mlb = self.mlbs[field]
            Y_pred_proba = model.predict_proba(X)  # list of arrays per class
            field_preds = []

            # Take probability for each class
            probs = np.array([clf[0][1] if clf.shape[1] > 1 else clf[0][0] for clf in Y_pred_proba])
            
            # Get top N indices
            top_indices = np.argpartition(probs, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

            # Map indices to class labels
            top_labels = [mlb.classes_[idx] for idx in top_indices]
            field_preds.append(top_labels)

            results[field] = field_preds

        return results
