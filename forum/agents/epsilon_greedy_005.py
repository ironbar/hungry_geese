"""
Copied from kaggle to see if it works better when parallelized
"""
import random
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation, Configuration, Action, adjacent_positions,
    min_distance,translate
)
from hungry_geese.agents import EpsilonAgent

class GreedyAgent:
    def __init__(self):
        self.last_action = None

    def __call__(self, observation: Observation, configuration: Configuration):
        rows, columns = configuration.rows, configuration.columns

        food = observation.food
        geese = observation.geese
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else random.choice([action for action in Action])
        self.last_action = action
        return action.name


base_agent = GreedyAgent()
epsilon_agent = EpsilonAgent(base_agent, epsilon=0.05)

def agent(obs, config):
    return epsilon_agent(obs, config)
