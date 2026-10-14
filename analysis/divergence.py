
"""Analyzes whether a small change in initial conditions causes agents
to follow different trajectories in the multi-agent simulation.
"""

import math
import matplotlib.pyplot as plt

from src.agents import Agent
from src.simulation import SimulationConfig, run_simulation


def calculate_divergence(baseline: list, comparison: list) -> list[tuple[int, float]]:
    """Measure positional differences between matching agents in two runs.

    This function supports the main research question by comparing the
    positions of corresponding agents at each timestep. It calculates
    the root-mean-square positional difference across all agents, so
    divergence can be tracked over the course of the experiment.
    """
    # Match each recorded position to its timestep and agent ID.
    # This makes sure each agent is compared with itself across both runs.
    baseline_by_state = {(timestep, agent_id): position
        for timestep, agent_id, position, velocity in baseline
    }
    comparison_by_state = {
        (timestep, agent_id): position
        for timestep, agent_id, position, velocity in comparison
    }
    if len(baseline_by_state) != len(baseline) or len(comparison_by_state) != len(comparison):
        raise ValueError("A trajectory contains duplicate timestep and agent ID records.")
    if not baseline_by_state:
        raise ValueError("Trajectories must contain at least one state.")
    # Both runs need to record the same agents at the same timesteps.
    if baseline_by_state.keys() != comparison_by_state.keys():
        raise ValueError("The two trajectories must record matching states.")

    timesteps = sorted({timestep for timestep, agent_id in baseline_by_state})
    agent_ids = sorted({agent_id for timestep, agent_id in baseline_by_state })
    if any((timestep, agent_id) not in baseline_by_state
           for timestep in timesteps for agent_id in agent_ids):
        raise ValueError("Each timestep must record the same agent IDs.")
    divergence = []

    for timestep in timesteps:
        squared_distances = []

        for agent_id in agent_ids:
            # Get the position of the same agent in each simulation.
            baseline_position = baseline_by_state[(timestep, agent_id)]
            stochastic_position = comparison_by_state[(timestep, agent_id)]
            # Calculate the distance between the two positions.

            dx = baseline_position[0] - stochastic_position[0]
            dy = baseline_position[1] - stochastic_position[1]

            squared_distance = dx**2 + dy**2
            squared_distances.append(squared_distance)

        # Summarize the differences across all agents at this timestep.
        mean_squared_distance = sum(squared_distances) / len(squared_distances)
        rmse = math.sqrt(mean_squared_distance)
        divergence.append((timestep, rmse))
    return divergence


def create_initial_state(config: SimulationConfig) -> list[Agent]:
    """Creates the same controlled starting arrangement for the experiment.

    A fixed grid gives the agents known starting positions and identical
    initial velocities. This makes it easier to isolate the effect of
    the planned perturbation instead of introducing random starting states.
    """
    # Place 20 agents in a grid inside the rectangular environment.
    agents = [
        Agent(
            agent_id=i,
            position=(2.0 + (i % 5) * 1.5,2.0 + (i // 5) * 1.5,),
            velocity=(0.1, 0.05),)
            for i in range(config.number_of_agents)
    ]
    return agents


def main() -> None:
    """Runs the initial-condition experiment and plots positional divergence.

    The baseline uses the original starting state. The perturbed run changes
    only one agent's initial x-position by a very small amount. Both runs
    use the same deterministic interaction rules and simulation parameters.
    The resulting graph shows whether the difference between their
    trajectories grows, shrinks, or stays approximately constant.
    """
    config = SimulationConfig(number_of_timesteps=500)

    # Create the baseline starting state for the first simulation.
    baseline_agents = create_initial_state(config)

    # Copy the starting state so the second run begins almost identically.
    # Only agent 0 receives a small change to its x-position.
    perturbation_size = 1e-5

    perturbed_agents = [
        Agent(agent_id=agent.id,
              position=(agent.position[0] +
                        (perturbation_size if agent.id == 0 else 0.0),
                        agent.position[1],),
                        velocity=agent.velocity,)
                        for agent in baseline_agents]

    # Run both experiments using the same rules and parameters.
    baseline_trajectory = run_simulation(baseline_agents, config)
    perturbed_trajectory = run_simulation(perturbed_agents, config)

    # Measure the difference between the two trajectories at each timestep.
    divergence = calculate_divergence(baseline_trajectory, perturbed_trajectory)
    timesteps = [timestep for timestep, value in divergence]
    values = [value for timestep, value in divergence]

    # Plot the positional difference to inspect sensitivity to initial conditions.
    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, values)

    plt.xlabel("Timestep")
    plt.ylabel("Positional divergence (RMSE)")
    plt.title("Sensitivity to Initial Conditions")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure so it can be used in the research report or presentation.
    from pathlib import Path

    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/initial_condition_divergence.png", dpi=200)
    plt.show()

    # Print the starting and ending differences for a quick numerical check.
    print(f"Initial divergence: {values[0]:.8f}")
    print(f"Final divergence:   {values[-1]:.8f}")


if __name__ == "__main__":
    main()
