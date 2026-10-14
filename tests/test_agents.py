import unittest

from src.agents import Agent


class AgentTests(unittest.TestCase):

    def test_initialization(self) -> None:
        agent = Agent(1, (2.0, 3.0), (0.5, -0.5))

        self.assertEqual(agent.id, 1)
        self.assertEqual(agent.get_position(), (2.0, 3.0))
        self.assertEqual(agent.get_velocity(), (0.5, -0.5))

    def test_get_position(self) -> None:
        agent = Agent(1, (2.0, 3.0), (0.5, -0.5))
        self.assertEqual(agent.get_position(), (2.0, 3.0))

    def test_get_velocity(self) -> None:
        agent = Agent(1, (2.0, 3.0), (0.1, 0.2))
        self.assertEqual(agent.get_velocity(), (0.1, 0.2))

    def test_attraction(self) -> None:
        agent = Agent(1, (0.0, 0.0), (0.0, 0.0))
        neighbor = Agent(2, (2.0, 0.0), (0.0, 0.0))

        parameters = {
            "attraction_strength": 0.5,
            "separation_strength": 0.0,
            "separation_radius": 0.5,
        }

        acceleration = agent.compute_acceleration( [neighbor], parameters)
        self.assertAlmostEqual(acceleration[0], 1.0)
        self.assertAlmostEqual(acceleration[1], 0.0)

    def test_separation(self) -> None:
        agent = Agent(1, (0.0, 0.0), (0.0, 0.0))
        neighbor = Agent(2, (0.25, 0.0), (0.0, 0.0))

        parameters = {
            "attraction_strength": 0.0,
            "separation_strength": 0.2,
            "separation_radius": 0.5,
        }

        acceleration = agent.compute_acceleration( [neighbor], parameters )
        self.assertAlmostEqual(acceleration[0], -0.2)
        self.assertAlmostEqual(acceleration[1], 0.0)

    def test_velocity_update(self) -> None:
        agent = Agent(1, (0.0, 0.0), (0.1, 0.2))
        agent.update_velocity((2.0, -1.0), 0.5)

        self.assertAlmostEqual(agent.get_velocity()[0], 1.1)
        self.assertAlmostEqual(agent.get_velocity()[1], -0.3)

    def test_position_update(self) -> None:
        agent = Agent(1, (1.0, 2.0), (0.2, -0.4))
        agent.update_position(0.5)

        self.assertAlmostEqual(agent.get_position()[0], 1.1)
        self.assertAlmostEqual(agent.get_position()[1], 1.8)

    def test_deterministic_updates(self) -> None:
        agent_one = Agent(1, (1.0, 2.0), (0.1, 0.2))
        agent_two = Agent(1, (1.0, 2.0), (0.1, 0.2))

        for agent in (agent_one, agent_two):
            agent.update_velocity((0.5, -0.25), 0.1)
            agent.update_position(0.1)

        self.assertEqual( agent_one.get_position(), agent_two.get_position())
        self.assertEqual(agent_one.get_velocity(), agent_two.get_velocity())


if __name__ == "__main__":
    unittest.main()
