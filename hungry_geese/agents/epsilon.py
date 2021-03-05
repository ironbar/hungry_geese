import random

from hungry_geese.utils import random_legal_action, opposite_action

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
