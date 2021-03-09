import random
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

class RandomPlus:
    def __init__(self):
        self.last_action = None

    def __call__(self):
        if self.last_action is not None:
            options = [action for action in Action if action != self.last_action.opposite()]
        else:
            options = [action for action in Action if action]

        action = random.choice(options)
        self.last_action = action
        return action.name

random_plus_agent = RandomPlus()

def agent(obs, config):
    return random_plus_agent()
