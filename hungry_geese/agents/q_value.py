import numpy as np

from hungry_geese.state import GameState, apply_all_simetries
from hungry_geese.heuristic import get_certain_death_mask, adapt_mask_to_3d_action
from hungry_geese.actions import get_action_from_relative_movement

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
        action = get_action_from_relative_movement(action_idx, self.previous_action)
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
    This version does not take risks if possible
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        safe_movements = np.arange(len(certain_death_mask))[certain_death_mask == 0]
        if safe_movements.size:
            return self._select_between_available_actions(safe_movements, q_value)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self._select_between_available_actions(risky_movements, q_value)

        return self._select_between_available_actions(np.arange(len(certain_death_mask)), q_value)

    def _select_between_available_actions(self, available_actions, q_value):
        q_value = q_value[available_actions]
        action_idx = available_actions[np.argmax(q_value)]
        return get_action_from_relative_movement(action_idx, self.previous_action)


class QValueSemiSafeAgent(QValueSafeAgent):
    """
    This version does not take risks if possible
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self._select_between_available_actions(risky_movements, q_value)

        return self._select_between_available_actions(np.arange(len(certain_death_mask)), q_value)


class QValueSafeMultiAgent(QValueSafeAgent):
    """
    Uses multiple models to create an stronger prediction
    """
    def __init__(self, models):
        self.models = models
        self.state = GameState()
        self.previous_action = 'NORTH'
        self.q_values = []

    def __call__(self, observation, configuration):
        # TODO: need to add function to handle first action
        board, features = self.state.update(observation, configuration)
        model_input = [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)]
        q_value = np.mean([model.predict_step(model_input)[0] for model in self.models], axis=0)
        self.q_values.append(q_value.copy())
        action = self.select_action(q_value, observation, configuration)
        self.previous_action = action
        self.state.add_action(action)
        return action


class QValueSafeAgentDataAugmentation(QValueSafeAgent):
    def _predict_q_value(self, board, features):
        data_augmented = apply_all_simetries(
            [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0), np.zeros((1, 3)), np.zeros((1, 3))])[:2]
        preds = self.model.predict_on_batch(data_augmented)
        fixed_preds = preds.copy()
        # horizontal simmetry
        fixed_preds[1::2, 0] = preds[1::2, 2]
        fixed_preds[1::2, 2] = preds[1::2, 0]
        q_value = np.mean(fixed_preds, axis=0)
        return q_value
