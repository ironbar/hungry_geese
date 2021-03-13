import numpy as np


def get_reward(current_observation, previous_observation, configuration, reward_name):
    if reward_name == 'sparse_reward':
        return get_sparse_reward(current_observation, previous_observation, configuration)
    elif reward_name.startswith('ranking_reward'):
        return get_ranking_reward(current_observation, reward_name)
    else:
        raise KeyError(reward_name)


def get_cumulative_reward(rewards, reward_name):
    if reward_name == 'sparse_reward':
        return np.cumsum(rewards[::-1])[::-1]
    elif reward_name.startswith('ranking_reward'):
        window_size = int(reward_name.split('_')[3])
        cumulative_reward = np.array(rewards)
        for idx in range(1, window_size):
            cumulative_reward[:-idx] += rewards[idx:]
        cumulative_reward[:-window_size] /= window_size
        for idx in range(2, window_size + 1):
            cumulative_reward[-idx] /= idx
        return cumulative_reward
    else:
        raise KeyError(reward_name)


def get_sparse_reward(current_observation, previous_observation, configuration):
    """ Computes the sparse reward for the previous action"""
    if current_observation['geese'][current_observation['index']]:
        is_terminal_step = current_observation['step'] == configuration['episodeSteps'] -1
        if is_terminal_step:
            return _get_terminal_sparse_reward(current_observation, previous_observation)
        else:
            # Give reward if some geese has died
            return get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])
    else:
        # Then the agent has died
        return -1


def _get_terminal_sparse_reward(current_observation, previous_observation):
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

def get_ranking_reward(current_observation, reward_name):
    geese_len = [len(goose) for goose in current_observation['geese']]
    goose_len = geese_len[current_observation['index']]
    if goose_len: # then it is alive
        reward = 0
        for idx, other_goose_len in enumerate(geese_len):
            if idx == current_observation['index']:
                continue
            if other_goose_len < goose_len:
                reward += 1
            elif other_goose_len == goose_len:
                reward += 0.5
        return reward
    else: # the agent has died
        return float(reward_name.split('_')[2])

