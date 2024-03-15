import pandas as pd
import unittest
from group_fairness import disparity_ratio, attributable_disparity
from individual_fairness import individial_unfairness, positive_reinforcement


def equal_relative(a, b, eps) -> bool:
    return abs(a - b) <= eps


class TestGroupFairnessMetrics(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1, 1, 1, 1, 0],
                "group": ["A", "B", "A", "A", "B", "B", "A", "A", "B"],
            }
        )
        self.epsilon = 1e-8

    def test_disparity_ratio(self):
        ratio = disparity_ratio(self.data, "outcome", "group", "A", "B")
        self.assertTrue(equal_relative(ratio, 0.625, self.epsilon))

    def test_attributable_disparity(self):
        disparity = attributable_disparity(self.data, "outcome", "group", "A", "B")
        self.assertTrue(equal_relative(disparity, -0.3, self.epsilon))


class TestIndividualFairnessMetrics(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1, 1, 1, 1, 0],
                "group": ["A", "B", "A", "A", "B", "B", "A", "A", "B"],
                "feature": ["C", "D", "C", "D", "C", "D", "D", "D", "C"],
            }
        )
        self.epsilon = 1e-8

    def test_positive_reinforcement(self):
        reinforcement = positive_reinforcement(
            self.data, "outcome", "group", "feature", lambda x: x == "D", "A", "B"
        )
        self.assertTrue(equal_relative(reinforcement, 0.5, self.epsilon))

    def test_individial_unfairness(self):
        unfairness = individial_unfairness(
            self.data,
            "outcome",
            "group",
            "feature",
            lambda x: x == "C",
            lambda x: x == "D",
            "A",
            "B",
        )
        print(unfairness)
        self.assertTrue(equal_relative(unfairness, 0.5, self.epsilon))


if __name__ == "__main__":
    unittest.main()
