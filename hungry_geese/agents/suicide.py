from hungry_geese.utils import opposite_action

class SuicideAgent():
    def __init__(self, action, steps_to_suicide):
        """
        Agent that will suicide after n steps if it is still alive, useful for testing
        """
        self.action = action
        self.steps = 0
        assert steps_to_suicide
        self.steps_to_suicide = steps_to_suicide

    def __call__(self, observation, configuration):
        if self.steps >= self.steps_to_suicide:
            return opposite_action(self.action)
        self.steps += 1
        return self.action

    def reset(self):
        self.steps = 0
