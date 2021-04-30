import numpy as np

from hungry_geese.heuristic import get_certain_death_mask, adapt_mask_to_3d_action
from hungry_geese.agents import QValueAgent
from hungry_geese.actions import get_action_from_relative_movement

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
        return self._sample_action_with_softmax(q_value, np.arange(len(q_value)))

    def _sample_action_with_softmax(self, prediction, choices):
        prediction -= np.mean(prediction) # to stabilize softmax
        probabilities = softmax(np.clip(prediction*self.scale, -100, 80)) # clip to avoid overflow and underflow
        action_idx = int(np.random.choice(choices, size=1, p=probabilities))
        action = get_action_from_relative_movement(action_idx, self.previous_action)
        return action


def softmax(x):
    output = np.exp(x)
    return output / np.sum(output)


class SoftmaxSafeAgent(SoftmaxAgent):
    def select_action(self, q_value, observation, configuration):
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        safe_movements = np.arange(len(certain_death_mask))[certain_death_mask == 0]
        if safe_movements.size:
            return self._sample_action_with_softmax(q_value[safe_movements], safe_movements)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self._sample_action_with_softmax(q_value[risky_movements], risky_movements)

        return self._sample_action_with_softmax(q_value, np.arange(len(certain_death_mask)))
