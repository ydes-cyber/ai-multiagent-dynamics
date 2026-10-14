"""Starter interfaces for basic trajectory comparisons."""

from .simulation import Trajectory


def trajectory_distance(
    trajectory_a: Trajectory, trajectory_b: Trajectory
) -> list[float]:
    """Return one distance value for each matching timestep."""
    # TODO: Define and calculate a distance between corresponding states.
    raise NotImplementedError


def validate_reproducibility(
    trajectory_a: Trajectory, trajectory_b: Trajectory, tolerance: float
) -> bool:
    """Check whether two deterministic runs agree within tolerance."""
    # TODO: Compare matching records using the supplied numerical tolerance.
    raise NotImplementedError
