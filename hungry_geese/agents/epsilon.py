import random
import numpy as np

from hungry_geese.utils import random_legal_action, opposite_action
from hungry_geese.heuristic import get_certain_death_mask, adapt_mask_to_3d_action
from hungry_geese.actions import get_action_from_relative_movement
from hungry_geese.agents.q_value import QValueSafeAgent


class EpsilonAgent():
    def __init__(self, agent, epsilon):
        """
        Epsilon-greedy agent that performs the action selected by the given and with probability
        epsilon selects a random action

        Parameters
        ----------
        agent : callable
            An agent as defined in kaggle competition
        epsilon : float
            Probability to choose a random action
        """
        self.agent = agent
        self.epsilon = epsilon
        self.previous_action = None

    def __call__(self, observation, configuration):
        action = self.agent(observation, configuration)
        if random.uniform(0, 1) < self.epsilon or self.previous_action is not None and action == opposite_action(self.previous_action):
            action = random_legal_action(self.previous_action)
            if hasattr(self.agent, 'update_previous_action'):
                self.agent.update_previous_action(action)
        self.previous_action = action
        return action

    def reset(self):
        self.previous_action = None
        if hasattr(self.agent, 'reset'):
            self.agent.reset()


class EpsilonSemiSafeAgent(QValueSafeAgent):
    """
    This version does not take risks if possible
    """
    def __init__(self, model, epsilon):
        """
        Parameters
        -----------
        model : keras Model
        epsilon : float
            Probability to choose a random action (0-1)
        """
        super().__init__(model)
        self.epsilon = epsilon

    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self.select_action_with_epsilon_greedy_policy(risky_movements, q_value)

        return self.select_action_with_epsilon_greedy_policy(np.arange(len(certain_death_mask)), q_value)

    def select_action_with_epsilon_greedy_policy(self, available_actions, q_value):
        if random.uniform(0, 1) < self.epsilon:
            action_idx = random.choice(available_actions)
            return get_action_from_relative_movement(action_idx, self.previous_action)

        return self._select_between_available_actions(available_actions, q_value)
