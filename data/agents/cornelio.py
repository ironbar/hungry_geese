


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
    return (boards, features) + tuple(data[2:])

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

n_units = 128
model = simple_model(
    conv_filters=[n_units, n_units, n_units, n_units],
    conv_activations=['relu', 'relu', 'relu', 'relu'],
    mlp_units=[n_units, n_units],
    mlp_activations=['relu', 'tanh'])

def get_weights():
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'

    return weights_b64

###############################################################################
# /mnt/hdd0/Kaggle/hungry_geese/models/46_train_from_zero_on_terminal_kill_and_grow_reward/12_just_train_and_evaluate_propagate_death_backwards_x128/epoch_00240.h5
# (1705, 1711)
###############################################################################

weights_b64 = get_weights()
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weights_b64))))
q_value_agent = QValueSemiSafeAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)