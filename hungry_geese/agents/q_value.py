import numpy as np

from hungry_geese.state import GameState
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX
from hungry_geese.utils import opposite_action
from hungry_geese.heuristic import get_certain_death_mask

class QValueAgent():
    def __init__(self, model):
        self.model = model
        self.state = GameState()
        self.previous_action = None
        self.q_values = []

    def __call__(self, observation, configuration):
        board, features = self.state.update(observation, configuration)
        q_value = np.array(self.model.predict_step([np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)])[0])
        self.q_values.append(q_value.copy())
        action = ACTIONS[self.select_action(q_value, observation, configuration)]
        self.previous_action = action
        self.state.add_action(action)
        return action

    def reset(self):
        self.state.reset()
        self.previous_action = None
        self.q_values = []

    def update_previous_action(self, previous_action):
        """ Allows to change previous action if an agent such as epsilon-greedy agent is playing"""
        self.previous_action = previous_action
        self.state.update_last_action(previous_action)

    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-3, len(q_value))
        if self.previous_action is not None:
            q_value[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 1e6
        return np.argmax(q_value)


class QValueSafeAgent(QValueAgent):
    """
    This version does not move into death positions
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        q_value -= np.floor(certain_death_mask)*1e2
        if self.previous_action is not None:
            q_value[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 1e6
        return np.argmax(q_value)


class QValueSafeMultiAgent(QValueSafeAgent):
    """
    Uses multiple models to create an stronger prediction
    """
    def __init__(self, models):
        self.models = models
        self.state = GameState()
        self.previous_action = None
        self.q_values = []

    def __call__(self, observation, configuration):
        board, features = self.state.update(observation, configuration)
        model_input = [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)]
        q_value = np.mean([model.predict_step(model_input)[0] for model in self.models], axis=0)
        self.q_values.append(q_value.copy())
        action = ACTIONS[self.select_action(q_value, observation, configuration)]
        self.previous_action = action
        self.state.add_action(action)
        return action