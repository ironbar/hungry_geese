


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
    weights_b64 = paste_weights_here

    return weights_b64

###############################################################################
# paste_model_path_here
# paste_model_score_here
###############################################################################

weights_b64 = get_weights()
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weights_b64))))
q_value_agent = QValueSafeAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)
