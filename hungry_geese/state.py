import numpy as np

class GameState():
    """
    Class that stores all observations and creates a game state made of the board
    and some features that are useful for planning
    """
    def __init__(self):
        self.history = []
        self.boards = []
        self.features = []
        self.configuration = None

    def update(self, observation, configuration):
        """
        Saves the observation to history and returns the state of the game
        """
        self.history.append(observation)
        if self.configuration is None:
            self.configuration = configuration

        self.features.append(self._compute_features(observation))

    def render(self, board):
        """
        Creates an rgb image to show the state of the board
        """
        pass

    def reset(self):
        """
        Deletes all data to be able to store a new episode
        """
        self.history = []
        self.boards = []
        self.features = []
        self.configuration = None

    def _compute_features(self, observation):
        features = np.zeros(2 + 2*len(observation['geese']) - 1)
        features[0] = get_steps_to_end(observation['step'], self.configuration['episodeSteps'])
        features[1] = get_steps_to_shrink(
            observation['step'], self.configuration['hunger_rate'])
        features[2] = get_steps_to_die(
            observation['step'], self.configuration['hunger_rate'],
            len(observation['geese'][observation['index']]))
        features[3:6] = [get_steps_to_die(
            observation['step'], self.configuration['hunger_rate'],
            len(goose)) for idx, goose in enumerate(observation['geese']) if idx != observation['index']]
        features[6:9] = [len(observation['geese'][observation['index']]) - \
            len(goose) for idx, goose in enumerate(observation['geese']) if idx != observation['index']]
        return features

def get_steps_to_shrink(step, hunger_rate):
    return hunger_rate - step % hunger_rate

def get_steps_to_die(step, hunger_rate, goose_len):
    if goose_len:
        return get_steps_to_shrink(step, hunger_rate) + (goose_len - 1)*hunger_rate
    else:
        return 0

def get_steps_to_end(step, episode_steps):
    return episode_steps - step
