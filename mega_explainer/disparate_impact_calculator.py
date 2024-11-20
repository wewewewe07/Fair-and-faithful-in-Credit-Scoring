import numpy as np
from collections import Counter

class DisparateImpactCalculator:
    def __init__(self, sorted_explanations, genders, top_n_features=5):
        """
        Initializes the DisparateImpactCalculator.

        Args:
            sorted_explanations (list): List of explanations, sorted by faithfulness.
            genders (list): List of genders corresponding to each explanation (e.g., ["male", "female"]).
            top_n_features (int): The number of top features to consider for each explanation.
        """
        self.sorted_explanations = sorted_explanations
        self.genders = genders
        self.top_n_features = top_n_features

        # Separate explanations into male and female groups
        self.sorted_explanations_male, self.sorted_explanations_female = self._separate_explanations()

    def _separate_explanations(self):
        """
        Separates the explanations into male and female groups based on gender labels.

        Gender mapping:
        - 0: Female
        - 1: Male

        Returns:
            tuple: (list of male explanations, list of female explanations)
            """
        male_explanations = [
            explanation for explanation, gender in zip(self.sorted_explanations, self.genders) if gender == 1
        ]
        female_explanations = [
            explanation for explanation, gender in zip(self.sorted_explanations, self.genders) if gender == 0
        ]
        return male_explanations, female_explanations

    def _extract_top_features(self, explanations):
        """
        Extracts the top N features from each explanation and counts occurrences across explanations.

        Args:
            explanations (list): List of explanations to analyze.

        Returns:
            Counter: Count of top features across all explanations.
        """
        feature_counts = Counter()
        for explanation in explanations:
            try:
                # Ensure explanation data is numeric
                explanation_data = np.array(explanation.list_exp, dtype=float)

                # Sort the features within each explanation by absolute importance and select the top N
                top_features = np.argsort(np.abs(explanation_data))[-self.top_n_features:]
                feature_counts.update(top_features)
            except ValueError:
                # Handle non-numeric data gracefully
                print(f"Non-numeric data encountered in explanation: {explanation.list_exp}")
                continue
        return feature_counts
    def calculate_disparate_impact(self):
        """
        Calculates the disparate impact ratio for top features across gender groups.

        Returns:
            dict: Disparate impact ratio for each feature.
        """
        # Extract and count top features for male and female groups
        male_feature_counts = self._extract_top_features(self.sorted_explanations_male)
        female_feature_counts = self._extract_top_features(self.sorted_explanations_female)

        # Total number of explanations in each group to calculate occurrence percentage
        total_male = len(self.sorted_explanations_male)
        total_female = len(self.sorted_explanations_female)

        # Calculate percentages and disparate impact ratio for each feature
        disparate_impact_ratios = {}
        all_features = set(male_feature_counts.keys()).union(female_feature_counts.keys())

        for feature in all_features:
            # Percentage of explanations where the feature is a top factor for each gender
            male_pct = male_feature_counts[feature] / total_male if total_male > 0 else 0
            female_pct = female_feature_counts[feature] / total_female if total_female > 0 else 0

            # Calculate disparate impact ratio
            if male_pct > 0:
                disparate_impact_ratio = female_pct / male_pct
            else:
                disparate_impact_ratio = np.nan  # Undefined if the feature never appears for males

            disparate_impact_ratios[feature] = disparate_impact_ratio

        return disparate_impact_ratios

    def interpret_disparate_impact(self, threshold=0.8):
        """
        Interprets the disparate impact ratios by identifying features with ratios below the threshold.

        Args:
            threshold (float): Threshold for identifying potential bias (default is 0.8).

        Returns:
            dict: Features with disparate impact ratios below the threshold, suggesting potential bias.
        """
        disparate_impact_ratios = self.calculate_disparate_impact()
        biased_features = {feature: ratio for feature, ratio in disparate_impact_ratios.items() if ratio < threshold}
        return biased_features
