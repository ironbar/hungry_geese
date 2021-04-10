import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions
import tensorflow.keras as keras
import base64
import bz2
import pickle
from itertools import permutations

"""
utils
"""

def opposite_action(action):
    action_to_opposite = {
        'NORTH': 'SOUTH',
        'EAST': 'WEST',
        'SOUTH': 'NORTH',
        'WEST': 'EAST',
    }
    return action_to_opposite[action]

"""
definitions
"""

ACTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

ACTION_TO_IDX = {
    'NORTH': 0,
    'EAST': 1,
    'SOUTH': 2,
    'WEST': 3,
}

"""
GameState
"""

class GameState():
    """
    Class that stores all observations and creates a game state made of the board
    and some features that are useful for planning
    """
    def __init__(self, egocentric_board=True, normalize_features=True, reward_name='sparse_reward'):
        self.history = []
        self.boards = []
        self.features = []
        self.rewards = []
        self.actions = []
        self.configuration = None
        self.egocentric_board = egocentric_board
        self.normalize_features = normalize_features
        self.reward_name = reward_name

    def update(self, observation, configuration):
        """
        Saves the observation to history and returns the state of the game
        """
        # if self.history:
        #     self.rewards.append(get_reward(observation, self.history[-1], configuration, self.reward_name))
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

    def reset(self):
        """
        Deletes all data to be able to store a new episode
        """
        self.history = []
        self.boards = []
        self.features = []
        self.rewards = []
        self.actions = []
        self.configuration = None

    # def prepare_data_for_training(self):
    #     """
    #     Returns
    #     --------
    #     boards: np.array
    #         Boards of the episode with shape (steps, 7, 11, 17) when 4 players
    #     features: np.array
    #         Features of the episode with shape (steps, 9) when 4 players
    #     actions: np.array
    #         Actions took during the episode with one hot encoding (steps, 4)
    #     rewards: np.array
    #         Cumulative reward received during the episode (steps,)
    #     """
    #     cumulative_reward = get_cumulative_reward(self.rewards, self.reward_name)
    #     actions = np.array(self.actions[:len(cumulative_reward)])
    #     for action, idx in ACTION_TO_IDX.items():
    #         actions[actions == action] = idx
    #     actions = tf.keras.utils.to_categorical(actions, num_classes=4)
    #     return [np.array(self.boards[:len(actions)], dtype=np.int8), np.array(self.features[:len(actions)]), actions, cumulative_reward]

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

def get_head_position(head, columns):
    row = head//columns
    col = head - row*columns
    return row, col

def vertical_simmetry(data):
    boards = data[0][:, ::-1].copy()
    return boards, data[1]

def horizontal_simmetry(data):
    boards = data[0][:, :, ::-1].copy()
    return boards, data[1]

def player_simmetry(data, new_positions):
    boards = data[0].copy()
    features = data[1].copy()
    for old_idx, new_idx in enumerate(new_positions):
        boards[:, :, :, 4*(new_idx+1):4*(new_idx+2)] = data[0][:, :, :, 4*(old_idx+1):4*(old_idx+2)]
        features[:, 3+new_idx] = data[1][:, 3+old_idx]
        features[:, 6+new_idx] = data[1][:, 6+old_idx]
    return boards, features

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

"""
QValueAgent
"""

class QValueAgent():
    def __init__(self, model):
        self.model = model
        self.state = GameState()
        self.previous_action = None
        self.q_values = []

    def __call__(self, observation, configuration):
        board, features = self.state.update(observation, configuration)
        q_value = self.predict_with_data_augmentation(board, features)
        self.q_values.append(q_value.copy())
        action = ACTIONS[self.select_action(q_value)]
        self.previous_action = action
        self.state.add_action(action)
        return action

    def predict_with_data_augmentation(self, board, features):
        data_augmented = apply_all_simetries([np.expand_dims(board, axis=0), np.expand_dims(features, axis=0)])
        preds = self.model.predict_on_batch(data_augmented)
        fixed_preds = preds.copy()
        # vertical simmetry
        fixed_preds[1::4, 0] = preds[1::4, 2]
        fixed_preds[1::4, 2] = preds[1::4, 0]
        # horizontal simmetry
        fixed_preds[2::4, 1] = preds[2::4, 3]
        fixed_preds[2::4, 3] = preds[2::4, 1]
        # both simetries
        fixed_preds[3::4, 1] = preds[3::4, 3]
        fixed_preds[3::4, 3] = preds[3::4, 1]
        fixed_preds[3::4, 0] = preds[3::4, 2]
        fixed_preds[3::4, 2] = preds[3::4, 0]
        q_value = np.mean(fixed_preds, axis=0)
        return q_value

    def reset(self):
        self.state.reset()
        self.previous_action = None
        self.q_values = []

    def update_previous_action(self, previous_action):
        """ Allows to change previous action if an agent such as epsilon-greedy agent is playing"""
        self.previous_action = previous_action
        self.state.update_last_action(previous_action)

    def select_action(self, q_value):
        q_value += np.random.uniform(0, 1e-3, len(q_value))
        if self.previous_action is not None:
            q_value[ACTION_TO_IDX[opposite_action(self.previous_action)]] -= 1e6
        return np.argmax(q_value)


"""
Model
"""
def simple_model(conv_filters, conv_activations, mlp_units, mlp_activations):
    board_input = keras.layers.Input((7, 11, 17), name='board_input')
    features_input = keras.layers.Input((9,), name='features_input')

    board_encoder = board_input
    for n_filters, activation in zip(conv_filters, conv_activations):
        board_encoder = keras.layers.Conv2D(n_filters, (3, 3), activation=activation, padding='valid')(board_encoder)
    board_encoder = keras.layers.Flatten()(board_encoder)

    output = keras.layers.concatenate([board_encoder, features_input])
    for units, activation in zip(mlp_units, mlp_activations):
        output = keras.layers.Dense(units, activation=activation)(output)
    output = keras.layers.Dense(4, activation='linear', name='action')(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model

model = simple_model(
    conv_filters=[128, 128, 128],
    conv_activations=['relu', 'relu', 'relu'],
    mlp_units=[128, 128],
    mlp_activations=['relu', 'tanh'])

def get_weights():

    return weights_b64

"""
/mnt/hdd0/Kaggle/hungry_geese/models/34_deep_q_learning/02_continue/epoch_0460.h5
Multi agent elo score: 1515
Single agent elo score: 1507
"""
weights_b64 = get_weights()
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weights_b64))))
q_value_agent = QValueAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)