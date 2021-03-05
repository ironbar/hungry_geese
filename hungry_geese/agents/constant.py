from hungry_geese.state import GameState

class ConstantAgent():
    def __init__(self, action='NORTH'):
        """
        Agent that always returns the same action, useful for testing
        """
        self.action = action

    def __call__(self, observation, configuration):
        return self.action

class ConstantAgentWithState():
    def __init__(self, action='NORTH'):
        """
        Agent that always returns the same action, useful for testing
        """
        self.action = action
        self.state = GameState()

    def __call__(self, observation, configuration):
        self.state.update(observation, configuration)
        self.state.add_action(self.action)
        return self.action

    def reset(self):
        self.state.reset()
