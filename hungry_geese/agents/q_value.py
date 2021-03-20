import numpy as np

from hungry_geese.state import GameState
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX
from hungry_geese.utils import opposite_action

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
        action = ACTIONS[self.select_action(q_value)]
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

    def select_action(self, q_value):
        q_value += np.random.uniform(0, 1e-3, len(q_value))
        if self.previous_action is not None:
            q_value[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 1e6
        return np.argmax(q_value)
