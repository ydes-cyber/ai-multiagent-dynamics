"""Starter tests for Environment. """

import unittest
from src.agents import Agent
from src.environment import Environment


class EnvironmentTests(unittest.TestCase):
    def test_initialization(self) -> None:
        """Check that width and height are stored correctly."""
        environment = Environment(100.0, 90.0)
        assert environment.width == 100.0
        assert environment.height == 90.0


    def test_is_inside(self) -> None:
        """Check interior, edge, and exterior positions."""
        environment = Environment(100.0, 90.0)
        assert environment.is_inside((50.0, 40.0)) is True
        assert environment.is_inside((-1.0, 40.0)) is False
        assert environment.is_inside((101.0, 40.0)) is False
        assert environment.is_inside((50.0, 91.0)) is False
        assert environment.is_inside((50.0, 90.0)) is True

    def test_handle_boundary(self) -> None:
        """Check deterministic reflection at each boundary."""
        environment = Environment(100.0, 90.0)
        agent = Agent(1, (-2.0, 40.0), (-3.0, 1.0))

        environment.handle_boundary(agent)

        assert agent.position == (2.0, 40.0)
        assert agent.velocity == (3.0, 1.0)

    def test_reflects_at_right_and_top_boundaries(self) -> None:
        """Check the remaining walls used by the rectangular environment."""
        environment = Environment(100.0, 90.0)
        agent = Agent(1, (102.0, 91.0), (3.0, 1.0))
        environment.handle_boundary(agent)
        self.assertEqual(agent.position, (98.0, 89.0))
        self.assertEqual(agent.velocity, (-3.0, -1.0))


