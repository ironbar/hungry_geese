import numpy as np

from hungry_geese.state import GameState
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX
from hungry_geese.utils import opposite_action
from hungry_geese.heuristic import get_certain_death_mask
from hungry_geese.agents import QValueAgent


class SoftmaxAgent(QValueAgent):
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
        super().__init__(model)
        self.scale = scale

    def select_action(self, q_value, observation, configuration):
        return self._sample_action_with_softmax(q_value)

    def _sample_action_with_softmax(self, prediction):
        prediction -= np.mean(prediction) # to stabilize softmax
        probabilities = softmax(np.clip(prediction*self.scale, -100, 80)) # clip to avoid overflow and underflow
        action_idx = int(np.random.choice(3, size=1, p=probabilities))
        action = ACTIONS[(action_idx - 1 + ACTION_TO_IDX[self.previous_action])%len(ACTIONS)]
        return action


def softmax(x):
    output = np.exp(x)
    return output / np.sum(output)


class SoftmaxSafeAgent(SoftmaxAgent):
    def select_action(self, q_value, observation, configuration):
        certain_death_mask = get_certain_death_mask(observation, configuration)
        # TODO: the mask needs to be adapted to 3d
        raise NotImplementedError()
        q_value -= certain_death_mask*1e3
        return self._sample_action_with_softmax(q_value)
