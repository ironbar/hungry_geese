"""
# Reward summary

## Sparse reward

This gives a reward each time that other agent dies. At the end of the
match it gives an extra reward depending on the ranking.

## Ranking reward

The reward is the current ranking of the agent on the match

## Clipped len reward

The reward is the difference between the leading goose and the agent, clipped.
If the agent is leading then is the difference with the second.

## Grow and kill reward

Growing gives reward and also the death of the other agents
"""

import numpy as np


def get_reward(current_observation, previous_observation, configuration, reward_name):
    if reward_name == 'sparse_reward':
        return get_sparse_reward(current_observation, previous_observation, configuration)
    elif reward_name.startswith('ranking_reward'):
        return get_ranking_reward(current_observation, reward_name)
    elif reward_name.startswith('clipped_len_reward'):
        return get_clipped_len_reward(current_observation, reward_name)
    elif reward_name.startswith('grow_and_kill_reward'):
        return get_grow_and_kill_reward(current_observation, previous_observation, reward_name)
    elif reward_name.startswith('just_survive'):
        get_just_survive_reward(current_observation, reward_name)
    else:
        raise KeyError(reward_name)


def get_cumulative_reward(rewards, reward_name):
    if reward_name == 'sparse_reward':
        return np.cumsum(rewards[::-1])[::-1]
    else:
        if reward_name.startswith('ranking_reward'):
            window_size = int(reward_name.split('_')[3])
        elif reward_name.startswith('clipped_len_reward'):
            window_size = _get_clipped_len_reward_params_from_name(reward_name)[1]
        elif reward_name.startswith('grow_and_kill_reward'):
            window_size = _get_grow_and_kill_reward_params_from_name(reward_name)[1]
        else:
            raise KeyError(reward_name)
        if window_size > len(rewards):
            window_size = len(rewards)
        cumulative_reward = np.array(rewards, dtype=np.float32)
        mask = np.ones_like(cumulative_reward)
        for idx in range(1, window_size):
            cumulative_reward[:-idx] += rewards[idx:]
            mask[:-idx] += 1
        cumulative_reward /= mask
        return cumulative_reward


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


def get_clipped_len_reward(current_observation, reward_name):
    death_reward, window, max_reward, min_reward = _get_clipped_len_reward_params_from_name(reward_name)
    geese_len = [len(goose) for goose in current_observation['geese']]
    goose_len = geese_len[current_observation['index']]
    if goose_len: # then it is alive
        max_len = max([geese_len[idx] for idx in range(len(geese_len)) if idx != current_observation['index']])
        len_diff = goose_len - max_len
        return np.clip(len_diff, min_reward, max_reward)
    else: # the agent has died
        return death_reward


def _get_clipped_len_reward_params_from_name(reward_name):
    death_reward, window, max_reward, min_reward = reward_name.split('_')[3:]
    return float(death_reward), int(window), float(max_reward), float(min_reward)


def get_grow_and_kill_reward(current_observation, previous_observation, reward_name):
    death_reward, window, max_reward, kill_reward = _get_grow_and_kill_reward_params_from_name(reward_name)
    geese_len = [len(goose) for goose in current_observation['geese']]
    goose_len = geese_len[current_observation['index']]
    if goose_len: # then it is alive
        kill_reward *= get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])
        max_len = max([geese_len[idx] for idx in range(len(geese_len)) if idx != current_observation['index']])
        len_diff = goose_len - max_len
        grow_reward = goose_len - len(previous_observation['geese'][current_observation['index']])
        if len_diff > max_reward or grow_reward < 0:
            grow_reward = 0
        return kill_reward + grow_reward
    else: # the agent has died
        return death_reward


def _get_grow_and_kill_reward_params_from_name(reward_name):
    """ grow_and_kill_reward_-1_8_3_1 """
    death_reward, window, max_reward, kill_reward = reward_name.split('_')[4:]
    return float(death_reward), int(window), float(max_reward), float(kill_reward)


def get_just_survive_reward(current_observation, reward_name):
    goose_len = len(current_observation['geese'][current_observation['index']])
    if goose_len: # then it is alive
        return 0
    else: # the agent has died
        death_reward = float(reward_name.split('_')[-1])
        return death_reward
