import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions

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
        if self.configuration is None:
            self.configuration = configuration

        self.features.append(self._compute_features(observation))
        self.boards.append(self._create_board(observation))
        self.history.append(observation)

    def render_board(self, board):
        """
        Creates an rgb image to show the state of the board, our agent is the red one
        """
        render = np.zeros(board.shape[:2] + (3,), dtype=np.uint8)
        idx_to_color = {
            0: np.array([85, 0, 0], dtype=np.uint8),
            1: np.array([0, 85, 0], dtype=np.uint8),
            2: np.array([0, 0, 85], dtype=np.uint8),
            3: np.array([0, 85, 85], dtype=np.uint8),
        }
        for idx in range(4):
            goose = board[:, :, idx*4] - board[:, :, idx*4+1] + board[:, :, idx*4+2]*2
            render += np.expand_dims(goose, axis=2).astype(np.uint8)*idx_to_color[idx]

        render += np.expand_dims(board[:, :, -1], axis=2).astype(np.uint8)*255
        return render

    def render_next_movements(self, board):
        """
        Creates an rgb image to show the avaible next movements, our agent is the red one
        """
        render = np.zeros(board.shape[:2] + (3,), dtype=np.uint8)
        idx_to_color = {
            0: np.array([85, 0, 0], dtype=np.uint8),
            1: np.array([0, 85, 0], dtype=np.uint8),
            2: np.array([0, 0, 85], dtype=np.uint8),
            3: np.array([0, 85, 85], dtype=np.uint8),
        }
        for idx in range(4):
            render += np.expand_dims(board[:, :, idx*4+3], axis=2).astype(np.uint8)*idx_to_color[idx]
        return render

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

    def _create_board(self, observation):
        """
        The board will have information about: head, body, tail and next movements
        Information will be separated in different channels so it is already high level
        """
        flat_board = np.zeros((self.configuration['rows']*self.configuration['columns'],
                              len(observation['geese'])*4+1))
        goose_order = [observation['index']] + [idx for idx in range(4) if idx != observation['index']]
        for idx in goose_order:
            goose = observation['geese'][idx]
            if goose:
                flat_board[goose[0], idx*4] = 1 # head
                flat_board[goose[-1], idx*4+1] = 1 # tail
                flat_board[goose, idx*4+2] = 1 # body
                next_movements = adjacent_positions(goose[0], rows=self.configuration['rows'], columns=self.configuration['columns'])
                flat_board[next_movements, idx*4+3] = 1 # next movements
                if self.history:
                    flat_board[self.history[-1]['geese'][idx][0], idx*4+3] = 0 # previous head position
        flat_board[observation['food'], -1] = 1
        board = np.reshape(flat_board, (self.configuration['rows'], self.configuration['columns'], len(observation['geese'])*4+1))
        #TODO: egocentric view
        return board

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
