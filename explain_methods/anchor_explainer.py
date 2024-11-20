from alibi.explainers import AnchorTabular
import numpy as np
import torch
import sys 
sys.path.append(r'c:\Users\Dell V3400\OneDrive\Tài liệu\machine learning\ML1 Project ')
from mega_explainer.base_explainer import BaseExplainer
class AnchorExplainer:
    def __init__(self, model, data, feature_names, categorical_names=None):
        """
        Initialize the AnchorExplainer.

        Args:
            model: The prediction model that takes in data as input and outputs predictions.
            data: The background data used to fit the explainer.
            feature_names: List of feature names for interpretability.
            categorical_names: Dictionary mapping column indices of categorical features to
                               lists of category names.
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.categorical_names = categorical_names

        # Initialize AnchorTabular
        self.anchor_explainer = AnchorTabular(model, feature_names=feature_names, categorical_names=categorical_names)
        self.anchor_explainer.fit(data, disc_perc=[25, 50, 75])

    def model_predict(self, data):
        """Wrap the model prediction to output classes only, needed for AnchorTabular."""
        return np.argmax(self.model(data), axis=1)

    def get_explanation(self, data, label=None):
        """
        Generate an explanation using the anchor method.

        Args:
            data: Instance to be explained (single row).
            label: (Unused in Anchor but included for compatibility).

        Returns:
            explanation: Tuple of (features with anchors, scores).
        """
        explanation = self.anchor_explainer.explain(data[0])
        anchors = explanation.anchor
        precision = explanation.precision  # This could be used as a "score"

        # Convert to a list of tuples for compatibility
        list_exp = [(self.feature_names[idx], 1) if idx in anchors else (self.feature_names[idx], 0) 
                    for idx in range(len(self.feature_names))]

        return list_exp, precision
