import numpy as np
import tensorflow.keras as keras
from itertools import permutations

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions

from hungry_geese.definitions import ACTION_TO_IDX
from hungry_geese.reward import get_reward, get_cumulative_reward

class GameState():
    """
    Class that stores all observations and creates a game state made of the board
    and some features that are useful for planning
    """
    def __init__(self, egocentric_board=True, normalize_features=True, reward_name='sparse_reward',
                 apply_reward_acumulation=True, forward_north_oriented=True, previous_action='NORTH'):
        """
        Parameters
        -----------
        egocentric_board : bool
            If true the head of the goose will be at the center of the board
        normalize_features : bool
            If true features will be normalized
        reward_name : str
            Name of the reward that we want to use
        apply_reward_acumulation : bool
            If true reward will be acumulated when returning train data
        forward_north_oriented : bool
            If true the board will be oriented so forward movement points north
        previous_action : str
            Name of the previous action, it will be used to orient the board forward north on first
            movement
        """
        self.history = []
        self.boards = []
        self.features = []
        self.rewards = []
        self.actions = [previous_action]
        self.configuration = None
        self.egocentric_board = egocentric_board
        self.normalize_features = normalize_features
        self.reward_name = reward_name
        self.apply_reward_acumulation = apply_reward_acumulation
        self.forward_north_oriented = forward_north_oriented

    def update(self, observation, configuration):
        """
        Saves the observation to history and returns the state of the game
        """
        if self.history:
            self.rewards.append(get_reward(observation, self.history[-1], configuration, self.reward_name))
        if self.configuration is None:
            self.configuration = configuration

        self.features.append(self._compute_features(observation))
        self.boards.append(self._create_board(observation))
        self.history.append(observation)
        return self.boards[-1], self.features[-1]

    def add_action(self, action):
        self.actions.append(action)

    def update_last_action(self, action):
        self.actions[-1] = action

    def get_last_action(self):
        return self.actions[-1]

    def render_board(self, board):
        """
        Creates an rgb image to show the state of the board, our agent is the red one
        """
        n_geese = board.shape[-1]//4
        render = np.zeros(board.shape[:2] + (3,), dtype=np.uint8)
        idx_to_color = {
            0: np.array([85, 0, 0], dtype=np.uint8),
            1: np.array([0, 85, 0], dtype=np.uint8),
            2: np.array([0, 0, 85], dtype=np.uint8),
            3: np.array([0, 85, 85], dtype=np.uint8),
        }
        for idx in range(n_geese):
            goose = board[:, :, idx*4] - board[:, :, idx*4+1] + board[:, :, idx*4+2]*2
            render += np.expand_dims(goose, axis=2).astype(np.uint8)*idx_to_color[idx]

        render += np.expand_dims(board[:, :, -1], axis=2).astype(np.uint8)*255
        return render

    def render_next_movements(self, board):
        """
        Creates an rgb image to show the avaible next movements, our agent is the red one
        """
        n_geese = board.shape[-1]//4
        render = np.zeros(board.shape[:2] + (3,), dtype=np.uint8)
        idx_to_color = {
            0: np.array([85, 0, 0], dtype=np.uint8),
            1: np.array([0, 85, 0], dtype=np.uint8),
            2: np.array([0, 0, 85], dtype=np.uint8),
            3: np.array([0, 85, 85], dtype=np.uint8),
        }
        for idx in range(n_geese):
            render += np.expand_dims(board[:, :, idx*4+3], axis=2).astype(np.uint8)*idx_to_color[idx]
        return render

    def reset(self, previous_action='NORTH'):
        """
        Deletes all data to be able to store a new episode

        Parameters
        ----------
        previous_action : str
            Name of the previous action, it will be used to orient the board forward north on first
            movement
        """
        self.history = []
        self.boards = []
        self.features = []
        self.rewards = []
        self.actions = [previous_action]
        self.configuration = None

    def prepare_data_for_training(self):
        """
        Returns
        --------
        boards: np.array
            Boards of the episode with shape (steps, 7, 11, 17) when 4 players
        features: np.array
            Features of the episode with shape (steps, 9) when 4 players
        actions: np.array
            Actions took during the episode with one hot encoding (steps, 4)
        rewards: np.array
            Cumulative reward received during the episode (steps,)
        """
        if self.apply_reward_acumulation:
            reward = get_cumulative_reward(self.rewards, self.reward_name)
        else:
            reward = self.rewards
        # TODO: update actions to be only of size 3 and take into account that an initial previous agent was added
        actions = np.array(self.actions[:len(reward)])
        for action, action_idx in ACTION_TO_IDX.items():
            actions[actions == action] = action_idx
        actions = keras.utils.to_categorical(actions, num_classes=4)
        return [np.array(self.boards[:len(actions)], dtype=np.int8), np.array(self.features[:len(actions)]), actions, reward]

    def _compute_features(self, observation):
        """
        Features convention
        -------------------
        0 -> step
        1 -> steps_to_shrink
        2 -> steps_to_die
        3:2 + n_geese -> step to die of the other goose
        2 + n_geese:1 + 2*n_geese -> len diff with the other goose
        """
        n_geese = len(observation['geese'])
        features = np.zeros(2 + 2*n_geese - 1, dtype=np.float32)
        features[0] = get_steps_to_end(observation['step'], self.configuration['episodeSteps'])
        features[1] = get_steps_to_shrink(
            observation['step'], self.configuration['hunger_rate'])
        features[2] = get_steps_to_die(
            observation['step'], self.configuration['hunger_rate'],
            len(observation['geese'][observation['index']]))
        features[3:2 + n_geese] = [get_steps_to_die(
            observation['step'], self.configuration['hunger_rate'],
            len(goose)) for idx, goose in enumerate(observation['geese']) if idx != observation['index']]
        features[2 + n_geese:1 + 2*n_geese] = [len(observation['geese'][observation['index']]) - \
            len(goose) for idx, goose in enumerate(observation['geese']) if idx != observation['index']]
        if self.normalize_features:
            features[0] /= self.configuration['episodeSteps']
            features[1] /= self.configuration['hunger_rate']
            features[2:2+n_geese] /= self.configuration['episodeSteps']
        return features

    def _create_board(self, observation):
        """
        The board will have information about: head, body, tail and next movements
        Information will be separated in different channels so it is already high level

        Channels convention
        --------------------
        There are 4*n_geese + 1 channels
        For each goose we have 4 channels: head, tail, body, next movements
        The last channel is the location of the food
        """
        n_geese = len(observation['geese'])
        flat_board = np.zeros((self.configuration['rows']*self.configuration['columns'], n_geese*4+1),
                              dtype=np.float32)
        goose_order = [observation['index']] + [idx for idx in range(n_geese) if idx != observation['index']]
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
        if self.egocentric_board:
            goose = observation['geese'][observation['index']]
            if goose:
                head_position = get_head_position(goose[0], self.configuration['columns'])
                board = make_board_egocentric(board, head_position)
        board = make_board_squared(board)
        if self.forward_north_oriented:
            board = make_board_forward_north_oriented(board, self.get_last_action())
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

def make_board_egocentric(board, head_position):
    """ Modifies the view of the board so the goose head is at the center"""
    row, col = head_position
    board = _center_board_rows(row, board)
    board = _center_board_cols(col, board)
    return board

def _center_board_rows(row, board):
    new_board = board.copy()
    half = board.shape[0]//2
    if row < half:
        offset = half - row
        new_board[offset:] = board[:-offset]
        new_board[:offset] = board[-offset:]
    elif row > half:
        offset = row - half
        new_board[:-offset] = board[offset:]
        new_board[-offset:] = board[:offset]
    return new_board

def _center_board_cols(col, board):
    new_board = board.copy()
    half = board.shape[1]//2
    if col < half:
        offset = half - col
        new_board[:, offset:] = board[:, :-offset]
        new_board[:, :offset] = board[:, -offset:]
    elif col > half:
        offset = col - half
        new_board[:, :-offset] = board[:, offset:]
        new_board[:, -offset:] = board[:, :offset]
    return new_board

def make_board_squared(board):
    """
    Creates a squared board by repeating the shortest dimension

    Typical board size is  (7, 11, 17), we want it to be (11, 11, 17)
    """
    if board.shape[0] > board.shape[1]:
        raise NotImplementedError('Currently is not supported bigger rows than cols: %s' % str(board.shape))
    side = np.max(board.shape[:2])
    squared_board = np.zeros((side, side, board.shape[2]), dtype=board.dtype)
    row_offset = (squared_board.shape[0] - board.shape[0])//2
    squared_board[row_offset:-row_offset] = board
    squared_board[:row_offset] = board[-row_offset:]
    squared_board[-row_offset:] = board[:row_offset]
    return squared_board

def make_board_forward_north_oriented(board, previous_action):
    """ Rotates the board so forward direction points north """
    action_idx = ACTION_TO_IDX[previous_action]
    if action_idx:
        board = np.rot90(board, k=action_idx)
    return board

def get_head_position(head, columns):
    row = head//columns
    col = head - row*columns
    return row, col

def vertical_simmetry(data):
    boards = data[0][:, ::-1].copy()
    actions = data[2].copy()
    # change north by south and viceversa
    actions[:, 0] = data[2][:, 2]
    actions[:, 2] = data[2][:, 0]
    return boards, data[1], actions, data[-1]

def horizontal_simmetry(data):
    boards = data[0][:, :, ::-1].copy()
    actions = data[2].copy()
    # change west by east and viceversa
    actions[:, 1] = data[2][:, 3]
    actions[:, 3] = data[2][:, 1]
    return boards, data[1], actions, data[-1]

def player_simmetry(data, new_positions):
    boards = data[0].copy()
    features = data[1].copy()
    for old_idx, new_idx in enumerate(new_positions):
        boards[:, :, :, 4*(new_idx+1):4*(new_idx+2)] = data[0][:, :, :, 4*(old_idx+1):4*(old_idx+2)]
        features[:, 3+new_idx] = data[1][:, 3+old_idx]
        features[:, 6+new_idx] = data[1][:, 6+old_idx]
    return boards, features, data[2], data[3]

def apply_all_simetries(data):
    all_data = []

    data_vertical = vertical_simmetry(data)
    data_horizontal = horizontal_simmetry(data)
    data_both = vertical_simmetry(horizontal_simmetry(data))
    all_permutations = list(permutations([0, 1, 2]))
    for new_positions in all_permutations:
        all_data.append(player_simmetry(data, new_positions))
        all_data.append(player_simmetry(data_vertical, new_positions))
        all_data.append(player_simmetry(data_horizontal, new_positions))
        all_data.append(player_simmetry(data_both, new_positions))
    return combine_data(all_data)

def combine_data(all_data):
    return [np.concatenate([_data[idx] for _data in all_data]) for idx in range(len(all_data[0]))]

def get_ohe_opposite_actions(actions):
    """
    Returns the opposite action given the actions in ohe encoding
    It simply rotates the matrix 2 positions to the right
    """
    opposite_actions = actions.copy()
    opposite_actions[:, :2] = actions[:, 2:]
    opposite_actions[:, 2:] = actions[:, :2]
    return opposite_actions
