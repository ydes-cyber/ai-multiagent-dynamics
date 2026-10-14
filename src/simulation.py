"""Configuration and starter entry points for the Version 1 simulation."""

from collections.abc import Sequence
from dataclasses import dataclass
from .agents import Agent, Vector2
from .environment import Environment
import math
import random


@dataclass
class SimulationConfig:
    """Fix simulation settings so experiments are comparable.

    noise_strength is the standard deviation of each x or y acceleration
    disturbance, in position units per model time unit squared. Zero leaves
    the deterministic update unchanged. random_seed reproduces nonzero noise.
    """
    number_of_agents: int = 20
    number_of_timesteps: int = 500
    timestep_size: float = 0.1
    environment_width: float = 10.0
    environment_height: float = 10.0
    interaction_radius: float = 3.0
    separation_radius: float = 0.5
    attraction_strength: float = 0.05
    separation_strength: float = 0.2
    damping: float = 0.1
    noise_strength: float = 0.0
    random_seed: int | None = None


# One record holds timestep, agent ID, position, and velocity.
StateRecord = tuple[int, int, Vector2, Vector2] # Definies what data gets recorded including: timestep, agentID, position, and velocity
Trajectory = list[StateRecord] ## Holds the record across the entire simulation


def validate_initial_state(
    initial_state: Sequence[Agent], config: SimulationConfig) -> None:
    """Checks whether the starting agents meet the experiment requirements. Prevents invalid starting conditions from corrupting your comparison"""
    # Check count, IDs, and finite values before comparing supplied states.
    if len(initial_state) != config.number_of_agents:
        raise ValueError("Initial state has the wrong number of agents")

    agent_ids = [agent.id for agent in initial_state]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Agent IDs must be unique")

    for agent in initial_state:
        if not all(math.isfinite(value) for value in agent.position):
            raise ValueError("Agent position doesn't contain finite numbers")
        if not all(math.isfinite(value) for value in agent.velocity):
            raise ValueError("Agent velocity doesn't contain finite numbers")

def run_simulation(
    initial_state: Sequence[Agent], config: SimulationConfig
) -> Trajectory:
    """Simulate agent movement and record every state over time.

    First, validate and copy the initial agents so the experiment does not
    alter the original starting state. Then calculate interactions,
    update velocities and positions, apply boundary behavior, and record
    every agent at every timestep for later statistical analysis.
    """
    validate_initial_state(initial_state, config)
    if not math.isfinite(config.noise_strength) or config.noise_strength < 0:
        raise ValueError("noise_strength must be a finite, nonnegative number")

    # Each run has its own generator. Zero noise leaves the original dynamics unchanged.
    random_generator = random.Random(config.random_seed) if config.noise_strength > 0 else None

    # Make independent Agent objects for this run.
    agents = [Agent(agent.id, agent.position, agent.velocity)
              for agent in initial_state
             ]

    environment = Environment(config.environment_width,config.environment_height,)

    parameters = {
        "attraction_strength": config.attraction_strength,
        "separation_strength": config.separation_strength,
        "separation_radius": config.separation_radius,
    }

    trajectory: Trajectory = []

    # Record the initial state before any movement occurs.
    for agent in agents:
        trajectory.append(
            (0, agent.id, agent.position, agent.velocity)
        )

    for timestep in range(1, config.number_of_timesteps + 1):
        # Calculate all accelerations from the same current state.
        accelerations = []

        for agent in agents:
            neighbors = [
                other for other in agents
                if other.id != agent.id
                and math.hypot(other.position[0] - agent.position[0],other.position[1] - agent.position[1],) <= config.interaction_radius
            ]

            acceleration = agent.compute_acceleration(neighbors, parameters)
            if random_generator is not None:
                # Draw separate x and y acceleration disturbances for every agent.
                acceleration = (
                    acceleration[0] + random_generator.gauss(0, config.noise_strength),
                    acceleration[1] + random_generator.gauss(0, config.noise_strength),
                )
            accelerations.append(acceleration)

        # Update velocities only after all accelerations are calculated.
        for agent, acceleration in zip(agents, accelerations):
            agent.update_velocity(acceleration, config.timestep_size)

        # Move agents and keep them inside the rectangle.
        for agent in agents:
            agent.update_position(config.timestep_size)
            environment.handle_boundary(agent)

        # Record the new state after movement.
        for agent in agents:
            trajectory.append((timestep, agent.id, agent.position, agent.velocity))

    return trajectory
