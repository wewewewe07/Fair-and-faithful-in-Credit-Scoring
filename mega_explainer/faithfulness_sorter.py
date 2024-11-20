# faithfulness_sorter.py

from dataclasses import dataclass
from typing import List

@dataclass
class Explanation:
    """Dataclass to store explanation details."""
    method_name: str
    explanation_data: list  # Explanation details for each feature
    faithfulness_score: float  # Faithfulness score of the explanation

class FaithfulnessSorter:
    """
    This class sorts explanations based on their faithfulness scores.
    """

    def __init__(self, explanations: List[Explanation]):
        """
        Initializes the sorter with explanations.
        
        Args:
            explanations: List of Explanation objects with faithfulness scores.
        """
        self.explanations = explanations

    def sort_by_faithfulness(self, descending: bool = True) -> List[Explanation]:
        """
        Sorts explanations based on faithfulness score.

        Args:
            descending: Whether to sort in descending order (highest faithfulness first).
        
        Returns:
            List of Explanation objects sorted by faithfulness score.
        """
        sorted_explanations = sorted(self.explanations, 
                                     key=lambda exp: exp.faithfulness_score, 
                                     reverse=descending)
        return sorted_explanations

    def get_top_explanations(self, top_n: int = 5) -> List[Explanation]:
        """
        Returns the top N explanations based on faithfulness score.

        Args:
            top_n: Number of top explanations to return.
        
        Returns:
            A list of the top N Explanation objects.
        """
        # Ensure the list is sorted first
        sorted_explanations = self.sort_by_faithfulness(descending=True)
        return sorted_explanations[:top_n]

# Example usage:
# explanations = [
#     Explanation(method_name="lime", explanation_data=[...], faithfulness_score=0.85),
#     Explanation(method_name="shap", explanation_data=[...], faithfulness_score=0.90),
#     Explanation(method_name="anchor", explanation_data=[...], faithfulness_score=0.80)
# ]
# sorter = FaithfulnessSorter(explanations)
# sorted_explanations = sorter.sort_by_faithfulness()
# top_explanation = sorter.get_top_explanations(top_n=1)
