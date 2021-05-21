


"""
utils.py
"""

"""
Functions commonly used in the challenge
"""
import os
import logging
import psutil
import time
import random
import tensorflow as tf


logger = logging.getLogger(__name__)

def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp

def opposite_action(action):
    action_to_opposite = {
        'NORTH': 'SOUTH',
        'EAST': 'WEST',
        'SOUTH': 'NORTH',
        'WEST': 'EAST',
    }
    return action_to_opposite[action]

def random_legal_action(previous_action):
    """ Returns a random action that is legal """
    if previous_action is not None:
        opposite = opposite_action(previous_action)
        options = [action for action in ACTIONS if action != opposite]
    else:
        options = ACTIONS
    action = random.choice(options)
    return action

def log_to_tensorboard(key, value, epoch, tensorboard_writer):
    with tensorboard_writer.as_default():
        tf.summary.scalar(key, value, step=epoch)

def log_configuration_to_tensorboard(configuration, tensorboard_writer, step=0):
    with tensorboard_writer.as_default():
        for key, value in configuration.items():
            tf.summary.text(key, str(value), step=step)

def get_ram_usage_and_available():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss/1e9
    stats = psutil.virtual_memory()  # returns a named tuple
    ram_available = getattr(stats, 'available')/1e9
    return ram_usage, ram_available


def get_cpu_usage():
    return psutil.cpu_percent()

def log_ram_usage():
    ram_usage, ram_available = get_ram_usage_and_available()
    #logger.info('ram_memory', used=round(ram_usage, 2), available=round(ram_available, 2))
    logger.info('ram_memory used: %.2f GB\t available: %.2f GB' % (ram_usage, ram_available))

def configure_logging(level=logging.DEBUG):
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


"""
definitions.py
"""

"""
Commonly used paths, labels, and other stuff

Examples from Coldstart challenge

DATASET_PATH = '/media/guillermo/Data/DrivenData/Cold_Start'
TRAIN_PATH = os.path.join(DATASET_PATH, 'data', 'train.csv')
TEST_PATH = os.path.join(DATASET_PATH, 'data', 'test.csv')
METADATA_PATH = os.path.join(DATASET_PATH, 'data', 'meta.csv')
SUBMISSION_PATH = os.path.join(DATASET_PATH, 'data', 'submission_format.csv')
LIBRARY_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""
import os
import pandas as pd
import yaml

ACTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

ACTION_TO_IDX = {
    'NORTH': 0,
    'EAST': 1,
    'SOUTH': 2,
    'WEST': 3,
}



"""
state.py
"""

import numpy as np
import tensorflow.keras as keras
from itertools import permutations

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions


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
            Actions took during the episode with one hot encoding (steps, 4) or steps(, 3) if
            forward_north_oriented is True
        rewards: np.array
            Cumulative reward received during the episode (steps,)
        """
        if self.apply_reward_acumulation:
            reward = get_cumulative_reward(self.rewards, self.reward_name)
        else:
            reward = self.rewards

        actions = np.array(self.actions[:len(reward) + 1])
        action_indices = np.zeros_like(actions, dtype=np.float32)
        for action, action_idx in ACTION_TO_IDX.items():
            action_indices[actions == action] = action_idx
        if self.forward_north_oriented:
            relative_movements = get_relative_movement_from_action_indices(action_indices)
            ohe_actions = keras.utils.to_categorical(relative_movements, num_classes=3)
        else: # then simply the action is the ohe of all the actions
            action_indices = action_indices[1:] # remove initial previous action
            ohe_actions = keras.utils.to_categorical(action_indices, num_classes=4)

        return [np.array(self.boards[:len(reward)], dtype=np.int8), np.array(self.features[:len(reward)]), ohe_actions, reward]

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
        for idx, goose_idx in enumerate(goose_order):
            goose = observation['geese'][goose_idx]
            if goose:
                flat_board[goose[0], idx*4] = 1 # head
                flat_board[goose[-1], idx*4+1] = 1 # tail
                flat_board[goose, idx*4+2] = 1 # body
                next_movements = adjacent_positions(goose[0], rows=self.configuration['rows'], columns=self.configuration['columns'])
                flat_board[next_movements, idx*4+3] = 1 # next movements
                if self.history:
                    flat_board[self.history[-1]['geese'][goose_idx][0], idx*4+3] = 0 # previous head position
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
    if actions.shape[1] == 4:
        # change west by east and viceversa
        actions[:, 1] = data[2][:, 3]
        actions[:, 3] = data[2][:, 1]
    elif actions.shape[1] == 3:
        # change turn left for turn right and viceversa
        actions[:, 0] = data[2][:, 2]
        actions[:, 2] = data[2][:, 0]
    else:
        raise NotImplementedError(actions.shape)
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
    all_permutations = list(permutations([0, 1, 2]))
    data_horizontal = horizontal_simmetry(data)
    if _is_vertical_simmetry_aplicable(data):
        data_vertical = vertical_simmetry(data)
        data_both = vertical_simmetry(horizontal_simmetry(data))
    for new_positions in all_permutations:
        all_data.append(player_simmetry(data, new_positions))
        all_data.append(player_simmetry(data_horizontal, new_positions))
        if _is_vertical_simmetry_aplicable(data):
            all_data.append(player_simmetry(data_vertical, new_positions))
            all_data.append(player_simmetry(data_both, new_positions))
    return combine_data(all_data)

def _is_vertical_simmetry_aplicable(data):
    return data[2].shape[1] == 4

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

def get_relative_movement_from_action_indices(action_indices):
    """
    Transforms the indices of the actions to relative movements being:

    - 0 turn left (NORTH -> WEST)
    - 1 forward (NORTH -> NORTH)
    - 2 turn right (NORTH -> EAST)
    """
    diff = action_indices[1:] - action_indices[:-1]
    movements = (diff + 1)%4
    return movements



"""
heuristic.py
"""

import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions

def get_certain_death_mask(observation, configuration):
    """
    Returns a mask for the actions that has ones on those actions that lead to a certain
    death, 0.5 on actions that may lead to death and 0 in the other cases

    array([ [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
            [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]])
    """
    geese = observation['geese']
    head_position = geese[observation['index']][0]
    future_positions = adjacent_positions(head_position, configuration['columns'], configuration['rows'])
    certain_death_mask = np.array([is_future_position_doomed(future_position, observation, configuration) for future_position in future_positions], dtype=np.float32)
    return certain_death_mask

def is_future_position_doomed(future_position, observation, configuration):
    """
    Returns
    -------
    0 if future position is safe
    1 if death is certain
    0.5 if death depends on other goose movements
    """
    is_doomed = 0
    for idx, goose in enumerate(observation['geese']):
        if not goose:
            continue
        # If the future position is on the body of the goose, then death is certain
        if future_position in goose[:-1]:
            return 1
        # The tail is special, if the goose is not going to grow it is safe
        if future_position == goose[-1] and idx != observation['index']:
            if is_food_around_head(goose, observation['food'], configuration):
                is_doomed = 0.5
        # Check if other goose may move into that position
        if idx != observation['index']:
            if is_food_around_head(goose, [future_position], configuration):
                is_doomed = 0.5
    return is_doomed

def is_food_around_head(goose, food, configuration):
    future_positions = adjacent_positions(goose[0], configuration['columns'], configuration['rows'])
    return any(food_position in future_positions for food_position in food)

def adapt_mask_to_3d_action(mask, previous_action):
    """
    Transforms the mask to fit the convention of: 0 turn left, 1 move forward and 2 turn right
    the input mask means: north, east, south, west
    """
    previous_action_to_indices = {
        'NORTH': [-1, 0, 1],
        'EAST': [0, 1, 2],
        'SOUTH': [1, 2, 3],
        'WEST': [2, 3, 0],
    }
    return mask[previous_action_to_indices[previous_action]]



"""
actions.py
"""

"""
Actions
"""

def get_action_from_relative_movement(relative_movement, previous_action):
    """
    Returns the action using the previous action and relative_movement(0,1,2s)
    """
    return ACTIONS[(relative_movement - 1 + ACTION_TO_IDX[previous_action])%len(ACTIONS)]



"""
q_value.py
"""

import numpy as np


class QValueAgent():
    """
    Agent that makes the action that maximizes the Q value
    """
    def __init__(self, model):
        self.model = model
        self.state = GameState()
        self.previous_action = 'NORTH'
        self.q_values = []
        self.is_first_action = True

    def __call__(self, observation, configuration):
        if self.is_first_action:
            q_value = self.first_action_q_value_estimation(observation, configuration)
            self.is_first_action = False
        else:
            board, features = self.state.update(observation, configuration)
            q_value = self._predict_q_value(board, features)
        self.q_values.append(q_value.copy())
        action = self.select_action(q_value, observation, configuration)
        self.previous_action = action
        self.state.add_action(action)
        return action

    def reset(self):
        self.state.reset()
        self.previous_action = 'NORTH'
        self.q_values = []
        self.is_first_action = True

    def update_previous_action(self, previous_action):
        """ Allows to change previous action if an agent such as epsilon-greedy agent is playing"""
        self.previous_action = previous_action
        self.state.update_last_action(previous_action)

    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        action_idx = np.argmax(q_value)
        action = get_action_from_relative_movement(action_idx, self.previous_action)
        return action

    def first_action_q_value_estimation(self, observation, configuration):
        """
        On first action we have to try both facing north and south
        """
        board, features = self.state.update(observation, configuration)
        q_value_north = self._predict_q_value(board, features)

        self.state.reset(previous_action='SOUTH')
        board, features = self.state.update(observation, configuration)
        q_value_south = self._predict_q_value(board, features)

        if np.max(q_value_south) > np.max(q_value_north):
            self.previous_action = 'SOUTH'
            q_value = q_value_south
        else:
            self.state.reset()
            self.state.update(observation, configuration)
            q_value = q_value_north
        return q_value

    def _predict_q_value(self, board, features):
        q_value = np.array(self.model.predict_step([np.expand_dims(board, axis=0),
                                                    np.expand_dims(features, axis=0)])[0])
        return q_value


class QValueSafeAgent(QValueAgent):
    """
    This version does not take risks if possible
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        safe_movements = np.arange(len(certain_death_mask))[certain_death_mask == 0]
        if safe_movements.size:
            return self._select_between_available_actions(safe_movements, q_value)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self._select_between_available_actions(risky_movements, q_value)

        return self._select_between_available_actions(np.arange(len(certain_death_mask)), q_value)

    def _select_between_available_actions(self, available_actions, q_value):
        q_value = q_value[available_actions]
        action_idx = available_actions[np.argmax(q_value)]
        return get_action_from_relative_movement(action_idx, self.previous_action)


class QValueSemiSafeAgent(QValueSafeAgent):
    """
    This version does not take risks if possible
    """
    def select_action(self, q_value, observation, configuration):
        q_value += np.random.uniform(0, 1e-5, len(q_value))
        certain_death_mask = get_certain_death_mask(observation, configuration)
        certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, self.previous_action)

        risky_movements = np.arange(len(certain_death_mask))[certain_death_mask < 1]
        if risky_movements.size:
            return self._select_between_available_actions(risky_movements, q_value)

        return self._select_between_available_actions(np.arange(len(certain_death_mask)), q_value)


class QValueSafeMultiAgent(QValueSafeAgent):
    """
    Uses multiple models to create an stronger prediction
    """
    def __init__(self, models):
        self.models = models
        self.state = GameState()
        self.previous_action = 'NORTH'
        self.q_values = []

    def __call__(self, observation, configuration):
        # TODO: need to add function to handle first action
        board, features = self.state.update(observation, configuration)
        model_input = [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)]
        q_value = np.mean([model.predict_step(model_input)[0] for model in self.models], axis=0)
        self.q_values.append(q_value.copy())
        action = self.select_action(q_value, observation, configuration)
        self.previous_action = action
        self.state.add_action(action)
        return action


class QValueSafeAgentDataAugmentation(QValueSafeAgent):
    def _predict_q_value(self, board, features):
        data_augmented = apply_all_simetries(
            [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0), np.zeros((1, 3)), np.zeros((1, 3))])[:2]
        preds = self.model.predict_on_batch(data_augmented)
        fixed_preds = preds.copy()
        # horizontal simmetry
        fixed_preds[1::2, 0] = preds[1::2, 2]
        fixed_preds[1::2, 2] = preds[1::2, 0]
        q_value = np.mean(fixed_preds, axis=0)
        return q_value


class QValueSemiSafeAgentDataAugmentation(QValueSemiSafeAgent):
    def _predict_q_value(self, board, features):
        data_augmented = apply_all_simetries(
            [np.expand_dims(board, axis=0), np.expand_dims(features, axis=0), np.zeros((1, 3)), np.zeros((1, 3))])[:2]
        preds = self.model.predict_on_batch(data_augmented)
        fixed_preds = preds.copy()
        # horizontal simmetry
        fixed_preds[1::2, 0] = preds[1::2, 2]
        fixed_preds[1::2, 2] = preds[1::2, 0]
        q_value = np.mean(fixed_preds, axis=0)
        return q_value


"""
model.py
"""

import tensorflow.keras as keras

N_ACTIONS = 3

def simple_model(conv_filters, conv_activations, mlp_units, mlp_activations):
    board_input, features_input = _create_model_input()

    board_encoder = board_input
    for n_filters, activation in zip(conv_filters, conv_activations):
        board_encoder = keras.layers.Conv2D(n_filters, (3, 3), activation=activation, padding='valid')(board_encoder)
    board_encoder = keras.layers.Flatten()(board_encoder)

    output = keras.layers.concatenate([board_encoder, features_input])
    for units, activation in zip(mlp_units, mlp_activations):
        output = keras.layers.Dense(units, activation=activation)(output)
    output = keras.layers.Dense(N_ACTIONS, activation='linear', name='action', use_bias=False)(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model


def _create_model_input():
    board_input = keras.layers.Input((11, 11, 17), name='board_input')
    features_input = keras.layers.Input((9,), name='features_input')
    return board_input, features_input


def create_model_for_training(model):
    input_mask = keras.layers.Input((N_ACTIONS,), name='input_mask')
    output = keras.backend.sum(input_mask*model.output, axis=-1)
    new_model = keras.models.Model(inputs=(model.inputs + [input_mask]), outputs=output)
    return new_model


def torus_model(torus_filters, summary_conv_filters, summary_conv_activations,
                feature_encoder_units, feature_encoder_activation,
                mlp_units, mlp_activations):
    """
    The idea is that the torus blocks extract features from the board, then we have some convolutional
    layers to summarize those features, concatenate with hand crafted features and a final mlp
    """
    board_input, features_input = _create_model_input()

    board_encoder = board_input
    for n_filters in torus_filters:
        board_encoder = torus_conv_bn_relu_block(board_encoder, n_filters)

    for n_filters, activation in zip(summary_conv_filters, summary_conv_activations):
        board_encoder = conv_bn_activation_block(board_encoder, n_filters, activation)
    board_encoder = keras.layers.Flatten()(board_encoder)

    features_encoder = dense_bn_activation_block(
        features_input, feature_encoder_units, feature_encoder_activation)

    output = keras.layers.concatenate([board_encoder, features_encoder])
    for units, activation in zip(mlp_units, mlp_activations):
        output = dense_bn_activation_block(output, units, activation)

    output = keras.layers.Dense(N_ACTIONS, activation='linear', name='action', use_bias=False)(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model


def torus_conv_bn_relu_block(x, n_filters):
    # import tensorflow.keras as keras
    # x = keras.layers.Lambda(
    #     lambda x: keras.backend.tile(x, n=(1, 3, 3, 1))[:, x.shape[1]-1:2*x.shape[1]+1, x.shape[2]-1:2*x.shape[2]+1,:],
    #     output_shape=lambda input_shape: (None, input_shape[1]+2, 3*input_shape[2]+2, input_shape[3]))(x)
    x = keras.backend.tile(x, n=(1, 3, 3, 1))[:, x.shape[1]-1:2*x.shape[1]+1, x.shape[2]-1:2*x.shape[2]+1,:]
    x = keras.layers.Conv2D(n_filters, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def conv_bn_activation_block(x, n_filters, activation):
    x = keras.layers.Conv2D(n_filters, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    return x

def dense_bn_activation_block(x, units, activation):
    x = keras.layers.Dense(units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    return x

###############################################################################
# Ending code
###############################################################################

import pickle
import bz2
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_reward(*args, **kwargs):
    return 0
get_cumulative_reward = get_reward


model = simple_model(
    conv_filters=[128, 128, 128, 128],
    conv_activations=['relu', 'relu', 'relu', 'relu'],
    mlp_units=[128, 128],
    mlp_activations=['relu', 'tanh'])

def get_weights():
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'

    return weights_b64

###############################################################################
# /mnt/hdd0/Kaggle/hungry_geese/models/40_study_effect_of_epsilon/04_epsilon_greedy_001/epoch_7920.h5
# paste_model_score_here
###############################################################################

weights_b64 = get_weights()
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weights_b64))))
q_value_agent = QValueSemiSafeAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)