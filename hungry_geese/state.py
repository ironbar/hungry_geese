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
        self.rewards = []
        self.configuration = None

    def update(self, observation, configuration):
        """
        Saves the observation to history and returns the state of the game
        """
        if self.history:
            self.rewards.append(get_reward(observation, self.history[-1], configuration))
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
        self.rewards = []
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

def get_reward(current_observation, previous_observation, configuration):
    """ Computes the reward for the previous action"""
    if current_observation['geese'][current_observation['index']]:
        is_terminal_step = current_observation['step'] == configuration['episodeSteps'] -1
        if is_terminal_step:
            return _get_terminal_reward(current_observation, previous_observation)
        else:
            # Give reward if some geese has died
            return get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])
    else:
        # Then the agent has died
        return -1

def _get_terminal_reward(current_observation, previous_observation):
    reward = get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])
    goose_len = len(current_observation['geese'][current_observation['index']])
    for idx, goose in enumerate(current_observation['geese']):
        if idx == current_observation['index']:
            continue
        other_goose_len = len(goose)
        if other_goose_len: # do not add rewards for already dead geese
            if goose_len > other_goose_len:
                reward += 1
            elif goose_len == other_goose_len:
                reward += 0.5
    return reward

def get_n_geese_alive(geese):
    return len([goose for goose in geese if goose])
