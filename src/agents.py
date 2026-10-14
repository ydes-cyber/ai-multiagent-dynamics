"""Starter interface for a deterministic agent in two dimensions."""

from collections.abc import Mapping, Sequence
import math

Vector2 = tuple[float, float]


class Agent:
    """An agent identified by an ID, position, and velocity."""

    id: int
    position: Vector2
    velocity: Vector2

    def __init__(self, agent_id: int, position: Vector2, velocity: Vector2) -> None:
        """Initialize this agent with the supplied ID, position, and velocity."""
        self.id = agent_id
        self.position = position
        self.velocity = velocity


    def get_position(self) -> Vector2:
        """Return this agent's current (x, y) position."""
        return self.position

    def get_velocity(self) -> Vector2:
        """Return this agent's current (vx, vy) velocity."""
        return self.velocity

    def compute_acceleration(
        self, neighbors: Sequence["Agent"], parameters: Mapping[str, float]
    ) -> Vector2:
        """Determine a deterministic acceleration from nearby agents."""


        # Get the strengths of attraction and separation from the experiment settings.
        # Attraction pulls the agent toward nearby agents, while separation pushes it away from agents that get too close.
        attraction_strength = parameters.get("attraction_strength", 0.0)
        separation_strength = parameters.get("separation_strength", 0.0)
        separation_radius = parameters.get("separation_radius", 0.5)

        # Start both effects at zero because the agent has not calculated any influence from its neighbors yet.
        attraction_x = 0.0
        attraction_y = 0.0
        separation_x = 0.0
        separation_y = 0.0

        if neighbors:
            for neighbor in neighbors:
                # Find the horizontal and vertical distance from this agent to each neighbor. These values tell us which direction the neighbor is relative to this agent.
                dx = neighbor.position[0] - self.position[0]
                dy = neighbor.position[1] - self.position[1]

                # Add each neighbor's displacement to can calculate the average direction toward the nearby group.
                attraction_x += dx
                attraction_y += dy

                 # Calculate the straight-line distance to the neighbor.
                distance = math.hypot(dx, dy)
                # If the neighbor is too close, calculate a push away from it.
                # Dividing by distance gives the direction without its originaldistance affecting the size of the push.

                if 0 < distance < separation_radius:
                    separation_x -= dx / distance
                    separation_y -= dy / distance

                # Average the attraction directions so the result reflects the overall position of the nearby group.
            attraction_x /= len(neighbors)
            attraction_y /= len(neighbors)

                # Combine attraction and separation, adjusting each effect using its configured strength. The result determines how velocity changes.
        acceleration_x = (attraction_strength * attraction_x + separation_strength * separation_x)
        acceleration_y = (attraction_strength * attraction_y + separation_strength * separation_y)
        return (acceleration_x, acceleration_y)

    def update_velocity(self, acceleration: Vector2, dt: float) -> None:
        """Update velocity using the supplied acceleration and timestep."""
        self.velocity =(self.velocity[0] + acceleration[0] * dt, self.velocity[1] + acceleration[1] * dt)
        return self.velocity

    def update_position(self, dt: float) -> None:
        """Update position using the current velocity and timestep."""
        self.position = (self.position[0] + self.velocity[0] *dt, self.position[1] + self.velocity[1] *dt)
        return self.position