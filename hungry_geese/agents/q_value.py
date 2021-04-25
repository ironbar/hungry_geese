import numpy as np

from hungry_geese.state import GameState
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX
from hungry_geese.utils import opposite_action
from hungry_geese.heuristic import get_certain_death_mask, adapt_mask_to_3d_action

class QValueAgent():
    """
    Agent that makes the action that maximizes the Q value
    """
    def __init__(self, model):
        self.model = model
        self.state = GameState()
        self.previous_action = 'NORTH'
        self.q_values = []
        self.is_first_action = True

    def __call__(self, observation, configuration):
        if self.is_first_action:
            q_value = self.first_action_q_value_estimation(observation, configuration)
            self.is_first_action = False
        else:
            board, features = self.state.update(observation, configuration)
            q_value = self._predict_q_value(board, features)
        self.q_values.append(q_value.copy())
        action = self.select_action(q_value, observation, configuration)
        self.previous_action = action
        self.state.add_action(action)
        return action

    def reset(self):
        self.state.reset()
        self.previous_action = 'NORTH'
        self.q_values = []
        self.is_first_action = True

    def update_previous_action(self, previous_action):
        """ Allows to change previous action if an agent such as epsilon-greedy agent is playing"""
        self.previous_action = previous_action
        self.state.update_last_action(previous_action)

    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        action_idx = np.argmax(q_value)
        action = ACTIONS[(action_idx - 1 + ACTION_TO_IDX[self.previous_action])%len(ACTIONS)]
        return action

    def first_action_q_value_estimation(self, observation, configuration):
        """
        On first action we have to try both facing north and south
        """
        board, features = self.state.update(observation, configuration)
        q_value_north = self._predict_q_value(board, features)

        self.state.reset(previous_action='SOUTH')
        board, features = self.state.update(observation, configuration)
        q_value_south = self._predict_q_value(board, features)

        if np.max(q_value_south) > np.max(q_value_north):
            self.previous_action = 'SOUTH'
            q_value = q_value_south
        else:
            self.state.reset()
            self.state.update(observation, configuration)
            q_value = q_value_north
        return q_value

    def _predict_q_value(self, board, features):
        q_value = np.array(self.model.predict_step([np.expand_dims(board, axis=0),
                                                    np.expand_dims(features, axis=0)])[0])
        return q_value


class QValueSafeAgent(QValueAgent):
    """
    This version does not move into death positions
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)
        q_value -= np.floor(certain_death_mask)*1e2
        action_idx = np.argmax(q_value)
        action = ACTIONS[(action_idx - 1 + ACTION_TO_IDX[self.previous_action])%len(ACTIONS)]
        return action


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
        raise NotImplementedError()
        board, features = self.state.update(observation, configuration)
        model_input = [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)]
        q_value = np.mean([model.predict_step(model_input)[0] for model in self.models], axis=0)
        self.q_values.append(q_value.copy())
        action = self.select_action(q_value, observation, configuration)
        self.previous_action = action
        self.state.add_action(action)
        return action
