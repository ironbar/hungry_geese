import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions

class GameState():
    """
    Class that stores all observations and creates a game state made of the board
    and some features that are useful for planning
    """
    def __init__(self, egocentric_board=True, normalize_features=True):
        self.history = []
        self.boards = []
        self.features = []
        self.rewards = []
        self.actions = []
        self.configuration = None
        self.egocentric_board = egocentric_board
        self.normalize_features = normalize_features

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