"""Check positional RMSE and trajectory matching."""

import math
import unittest

from analysis.divergence import calculate_divergence


class DivergenceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = [
            (0, 0, (1.0, 1.0), (0.0, 0.0)),
            (0, 1, (2.0, 2.0), (0.0, 0.0)),
            (1, 0, (1.5, 1.0), (0.0, 0.0)),
            (1, 1, (2.5, 2.0), (0.0, 0.0)),
        ]

    def test_self_divergence_is_zero(self):
        self.assertEqual(calculate_divergence(self.baseline, self.baseline),
                         [(0, 0.0), (1, 0.0)])

    def test_one_agent_perturbation_has_expected_initial_rmse(self):
        perturbation = 1e-5
        comparison = list(self.baseline)
        comparison[0] = (0, 0, (1.0 + perturbation, 1.0), (0.0, 0.0))
        self.assertAlmostEqual(calculate_divergence(self.baseline, comparison)[0][1],
                               perturbation / math.sqrt(2))

    def test_record_order_does_not_change_matching(self):
        self.assertEqual(calculate_divergence(self.baseline, list(reversed(self.baseline))),
                         [(0, 0.0), (1, 0.0)])

    def test_missing_and_duplicate_states_raise_errors(self):
        with self.assertRaisesRegex(ValueError, "matching states"):
            calculate_divergence(self.baseline, self.baseline[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calculate_divergence(self.baseline, self.baseline + [self.baseline[0]])
        with self.assertRaisesRegex(ValueError, "same agent IDs"):
            calculate_divergence(self.baseline[:-1], self.baseline[:-1])
