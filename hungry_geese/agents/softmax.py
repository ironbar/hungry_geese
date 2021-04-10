import numpy as np

from hungry_geese.state import GameState
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX
from hungry_geese.utils import opposite_action
from hungry_geese.heuristic import get_certain_death_mask

class SoftmaxAgent():
    """
    Random agent that performs a softmax policy over the q value or other model prediction
    """
    def __init__(self, model, scale=1):
        """
        Parameters
        ----------
        model : keras model
        scale : float
            Parameter that scales the softmax, the lower the softer the distribution
        """
        self.model = model
        self.state = GameState()
        self.previous_action = None
        self.model_predictions = []
        self.scale = scale

    def __call__(self, observation, configuration):
        board, features = self.state.update(observation, configuration)
        prediction = np.array(self.model.predict_step([np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)])[0])
        self.model_predictions.append(prediction.copy())
        action = ACTIONS[self.select_action(prediction, observation, configuration)]
        self.previous_action = action
        self.state.add_action(action)
        return action

    def reset(self):
        self.state.reset()
        self.previous_action = None
        self.model_predictions = []

    def update_previous_action(self, previous_action):
        """ Allows to change previous action if an agent such as epsilon-greedy agent is playing"""
        self.previous_action = previous_action
        self.state.update_last_action(previous_action)

    def select_action(self, prediction, observation, configuration):
        if self.previous_action is not None:
            prediction[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 1e20
        return int(np.random.choice(4, size=1, p=softmax(prediction*self.scale)))

def softmax(x):
    output = np.exp(x)
    return output / np.sum(output)

class SoftmaxSafeAgent(SoftmaxAgent):
    def select_action(self, prediction, observation, configuration):
        if self.previous_action is not None:
            prediction[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 2e10
        certain_death_mask = get_certain_death_mask(observation, configuration)
        prediction -= certain_death_mask*1e10
        return int(np.random.choice(4, size=1, p=softmax(prediction*self.scale)))