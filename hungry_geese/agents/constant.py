class ConstantAgent():
    def __init__(self, action='NORTH'):
        """
        Agent that always returns the same action, useful for testing
        """
        self.action = action

    def __call__(self, observation, configuration):
        return self.action