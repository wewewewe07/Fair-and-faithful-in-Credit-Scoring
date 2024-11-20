import numpy as np

class FairFaithfulSelector:
    def __init__(self, sorted_explanations, disparate_impact_ratios, fairness_threshold=0.8):
        """
        Initializes the FairFaithfulSelector.

        Args:
            sorted_explanations (list): List of Explanation objects, sorted by faithfulness (highest first).
            disparate_impact_ratios (dict): Disparate impact ratios for features across gender groups.
            fairness_threshold (float): Minimum fairness ratio threshold for disparate impact.
        """
        self.sorted_explanations = sorted_explanations
        self.disparate_impact_ratios = disparate_impact_ratios
        self.fairness_threshold = fairness_threshold

    def _check_fairness(self, explanation_data):
        """
        Checks if an explanation is fair by ensuring that all features in the explanation
        meet the fairness threshold.

        Args:
            explanation_data (np.ndarray): Explanation feature importance scores or feature indices.

        Returns:
            bool: True if the explanation meets fairness criteria, False otherwise.
        """
        # Check fairness for each feature in the explanation
        for feature in explanation_data:
            # Ensure the feature has a disparate impact ratio meeting the fairness threshold
            if feature in self.disparate_impact_ratios:
                if self.disparate_impact_ratios[feature] < self.fairness_threshold:
                    return False
        return True
    
    def find_most_fair_and_faithful(self):
        # Iterate through explanations (already sorted by faithfulness, highest first)
        for mega_explanation in self.sorted_explanations:
            # mega_explanation.list_exp contains the explanation data
            # mega_explanation.score contains the faithfulness score
            if self._check_fairness(mega_explanation.list_exp):
                return {
                    "method_name": mega_explanation.best_explanation_type,
                    "faithfulness_score": mega_explanation.score,
                    "explanation_data": mega_explanation.list_exp,
                    "is_fair": True
                }
        return None