
"""Run the baseline simulation and plot agent trajectories."""

import matplotlib.pyplot as plt

from src.agents import Agent
from src.simulation import SimulationConfig, run_simulation


def plot_trajectories(trajectory, config):
    """Plot recorded agent paths so the baseline motion can be inspected."""
    paths = {}
    for timestep, agent_id, position, velocity in trajectory:
        paths.setdefault(agent_id, []).append(position)

    fig, ax = plt.subplots(figsize=(8, 8))
    for agent_id, positions in paths.items():
        x_values = [position[0] for position in positions]
        y_values = [position[1] for position in positions]
        ax.plot(x_values, y_values, linewidth=1, label=f"Agent {agent_id}")
        ax.scatter(x_values[0], y_values[0], s=20)

    ax.set_xlim(0, config.environment_width)
    ax.set_ylim(0, config.environment_height)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Deterministic Multi-Agent Simulation")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    """Run a simulation and plot every agent's path through the environment."""
    config = SimulationConfig(number_of_timesteps=500)

    agents = [
        Agent(
            agent_id=i,
            position=(
                2.0 + (i % 5) * 1.5,
                2.0 + (i // 5) * 1.5,
            ),
            velocity=(0.1, 0.05),
        )
        for i in range(config.number_of_agents)
    ]

    trajectory = run_simulation(agents, config)

    fig = plot_trajectories(trajectory, config)
    plt.show()


if __name__ == "__main__":
    main()
