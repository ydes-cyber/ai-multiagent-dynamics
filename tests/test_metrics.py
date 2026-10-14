"""Tests for trajectory distance and reproducibility metrics."""

import unittest

from src.metrics import trajectory_distance, validate_reproducibility


class MetricsTests(unittest.TestCase):

    def test_identical_trajectories_have_zero_distance(self) -> None:
        trajectory_one = [
            (0, 1, (1.0, 2.0), (0.1, 0.2)),
            (1, 1, (1.1, 2.2), (0.1, 0.2)),]

        trajectory_two = [
            (0, 1, (1.0, 2.0), (0.1, 0.2)),
            (1, 1, (1.1, 2.2), (0.1, 0.2)),]

        distance = trajectory_distance(trajectory_one, trajectory_two)
        self.assertAlmostEqual(distance, 0.0)

    def test_different_trajectories_have_nonzero_distance(self) -> None:
        trajectory_one = [(0, 1, (1.0, 2.0), (0.1, 0.2)),]
        trajectory_two = [(0, 1, (4.0, 6.0), (0.1, 0.2)),]

        distance = trajectory_distance(trajectory_one, trajectory_two)
        self.assertGreater(distance, 0.0)

    def test_reproducibility_within_tolerance(self) -> None:
        self.assertTrue(validate_reproducibility(1.0, 1.00001, 0.001))

    def test_reproducibility_outside_tolerance(self) -> None:
        self.assertFalse(validate_reproducibility(1.0, 1.1, 0.001))

if __name__ == "__main__":
    unittest.main()