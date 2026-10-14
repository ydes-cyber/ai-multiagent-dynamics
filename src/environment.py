"""Starter interface for a rectangular two-dimensional environment."""

from .agents import Agent, Vector2


class Environment:
    """A continuous rectangle with deterministic boundary behavior."""

    width: float
    height: float

    def __init__(self, width: float, height: float) -> None:
        """Establishes the size of the simulated world. Currently is set to be a rectangle keeps the environment consistent across the experiment. If you change the size betweent the runs that will become another experimental variable"""
        self.height = height
        self.width = width

    def is_inside(self, position: Vector2) -> bool:
        """Determine whether a position lies within the rectangle."""
        if position[0] < 0 or position[0] > self.width:
            return False
        if position[1] < 0 or position[1] > self.height:
            return False
        return True

    def handle_boundary(self, agent: Agent) -> None:
        """Keeps the agent inside using deterministic reflecting boundaries for every experiment."""
       # Reverse the position and velocity of the agent if it goes out of bounds
        agent_bounds = self.is_inside(agent.position)
        if not agent_bounds:
            if agent.position[0] < 0: #left wall
                agent.position = (-agent.position[0], agent.position[1])
                agent.velocity = (-agent.velocity[0], agent.velocity[1])

            if agent.position[0] > self.width: #Right wall
                agent.position = (2 * self.width - agent.position[0], agent.position[1])
                agent.velocity = (-agent.velocity[0], agent.velocity[1])

            if agent.position[1] < 0: #Bottom wall
                agent.position = (agent.position[0], -agent.position[1])
                agent.velocity = (agent.velocity[0], -agent.velocity[1])
            if agent.position[1] > self.height: #Top wall
                agent.position = (agent.position[0], 2 * self.height - agent.position[1]) #set the position to be inside the boundary
                agent.velocity = (agent.velocity[0], -agent.velocity[1])
