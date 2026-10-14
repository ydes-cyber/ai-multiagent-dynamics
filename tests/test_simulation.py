"""Check the simulation conditions used by the research demo."""

import unittest

from src.agents import Agent
from src.simulation import SimulationConfig, run_simulation, validate_initial_state


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(number_of_agents=3, number_of_timesteps=4)
        self.agents = [
            Agent(0, (2.0, 2.0), (0.1, 0.05)),
            Agent(1, (3.0, 2.0), (0.1, 0.05)),
            Agent(2, (2.0, 3.0), (0.1, 0.05)),
        ]

    def test_records_every_agent_at_every_timestep(self):
        trajectory = run_simulation(self.agents, self.config)
        self.assertEqual(len(trajectory), 3 * 5)
        for timestep in range(5):
            self.assertEqual(
                {agent_id for step, agent_id, position, velocity in trajectory
                 if step == timestep},
                {0, 1, 2},
            )

    def test_does_not_mutate_initial_agents(self):
        initial_values = [(agent.position, agent.velocity) for agent in self.agents]
        run_simulation(self.agents, self.config)
        self.assertEqual(
            [(agent.position, agent.velocity) for agent in self.agents],
            initial_values,
        )

    def test_deterministic_runs_match_and_zero_noise_preserves_them(self):
        baseline = run_simulation(self.agents, self.config)
        self.assertEqual(baseline, run_simulation(self.agents, self.config))
        zero_noise = SimulationConfig(number_of_agents=3, number_of_timesteps=4,
                                      noise_strength=0.0, random_seed=42)
        self.assertEqual(baseline, run_simulation(self.agents, zero_noise))

    def test_fixed_seed_reproduces_stochastic_trajectory(self):
        config = SimulationConfig(number_of_agents=3, number_of_timesteps=4,
                                  noise_strength=0.1, random_seed=42)
        first = run_simulation(self.agents, config)
        self.assertEqual(first, run_simulation(self.agents, config))
        self.assertNotEqual(first, run_simulation(self.agents, self.config))

    def test_different_seeds_change_stochastic_trajectory(self):
        first_config = SimulationConfig(number_of_agents=3, number_of_timesteps=4,
                                        noise_strength=0.1, random_seed=42)
        second_config = SimulationConfig(number_of_agents=3, number_of_timesteps=4,
                                         noise_strength=0.1, random_seed=43)
        self.assertNotEqual(run_simulation(self.agents, first_config),
                            run_simulation(self.agents, second_config))

    def test_noise_draws_differ_between_agents(self):
        config = SimulationConfig(number_of_agents=3, number_of_timesteps=1,
                                  attraction_strength=0.0, separation_strength=0.0,
                                  noise_strength=0.1, random_seed=42)
        stationary_agents = [Agent(agent.id, agent.position, (0.0, 0.0))
                             for agent in self.agents]
        trajectory = run_simulation(stationary_agents, config)
        x_velocities = [velocity[0] for step, agent_id, position, velocity in trajectory
                        if step == 1]
        self.assertEqual(len(set(x_velocities)), 3)

    def test_invalid_noise_strength_is_rejected(self):
        config = SimulationConfig(number_of_agents=3, noise_strength=-0.1)
        with self.assertRaisesRegex(ValueError, "noise_strength"):
            run_simulation(self.agents, config)

    def test_duplicate_initial_ids_are_rejected(self):
        agents = [self.agents[0], self.agents[0], self.agents[2]]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_initial_state(agents, self.config)
