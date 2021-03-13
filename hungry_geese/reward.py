import numpy as np

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

def get_cumulative_reward(rewards):
    return np.cumsum(rewards[::-1])[::-1]
