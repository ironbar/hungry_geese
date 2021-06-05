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

## Terminal, kill and grow reward (terminal_kill_and_grow_reward)

- Terminal reward: this will be given if the agent dies or reaches the terminal state. It will be
proportional to the ranking of the agent, f.e. 5, -5, -10, -15. This is a difference with current
reward that was not giving reward on the terminal state. This way the agent will not find any
difference on dying on step 190 or reaching step 200 on second position if only two goose are left.
It will incentivate taking more risks for winning.
- Kill reward. this is already present on current reward and I think is a good incentive. Sometimes
the other agents will die by themselves, but if the agent learns to kill them that would be a good
ability. We could give for example 2 for each killed agent.
- Eat reward. I also think that encourage growing eases the learning.

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
        return get_just_survive_reward(current_observation, reward_name)
    elif reward_name.startswith('terminal_kill_and_grow_reward'):
        return get_terminal_kill_and_grow_reward(current_observation, previous_observation, reward_name, configuration)
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
        is_terminal_step = current_observation['step'] == configuration['episodeSteps'] - 1
        if is_terminal_step:
            return _get_terminal_sparse_reward(current_observation, previous_observation)
        else:
            # Give reward if some geese has died
            return get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])
    else:
        # Then the agent has died
        return -1


def _get_terminal_sparse_reward(current_observation, previous_observation):
    """
    Returns a reward between 0 and 3, where 0 means the agent is in the last position and 3 means
    it is the winner. It gives 1 point for each smaller or death agent, and 0.5 for each agent
    of the same size
    """
    current_geese_len = _get_geese_len(current_observation)
    previous_geese_len = _get_geese_len(previous_observation)
    goose_idx = current_observation['index']
    goose_len = current_geese_len[goose_idx]
    reward = 0
    for idx, (current_len, previous_len) in enumerate(zip(current_geese_len, previous_geese_len)):
        if idx == goose_idx:
            continue
        if not previous_len:
            reward += 1
            continue
        if goose_len > current_len:
            reward += 1
        elif goose_len == current_len:
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
        return 0.
    else: # the agent has died
        death_reward = float(reward_name.split('_')[-1])
        return death_reward


def get_terminal_kill_and_grow_reward(current_observation, previous_observation, reward_name, configuration):
    terminal_reward_scale, kill_reward, grow_reward = _get_terminal_kill_and_grow_reward_params_from_name(reward_name)
    if is_terminal_state(current_observation, configuration):
        terminal_reward = _get_terminal_sparse_reward(current_observation, previous_observation)
        terminal_reward -= 2.5 # this only gives positive reward for winning, [-2.5, 0.5]
        terminal_reward *= terminal_reward_scale
        return terminal_reward
    else:
        reward = kill_reward * _get_killed_geese(current_observation, previous_observation)
        reward += grow_reward * _get_goose_growth(current_observation, previous_observation, configuration)
        return reward


def is_terminal_state(current_observation, configuration):
    if _is_goose_death(current_observation):
        return True
    if _is_final_state(current_observation, configuration):
        return True
    if _are_all_other_goose_death(current_observation):
        return True
    return False


def _is_goose_death(observation):
    geese_len = _get_geese_len(observation)
    goose_len = geese_len[observation['index']]
    return not goose_len


def _are_all_other_goose_death(observation):
    geese_len = _get_geese_len(observation)
    for idx, goose_len in enumerate(geese_len):
        if idx == observation['index']:
            continue
        if goose_len:
            return False
    return True


def _get_geese_len(observation):
    geese_len = [len(goose) for goose in observation['geese']]
    return geese_len


def _is_final_state(observation, configuration):
    return observation['step'] == configuration['episodeSteps'] - 1


def _get_killed_geese(current_observation, previous_observation):
    """ Computes how many geese were killed between observations """
    return get_n_geese_alive(previous_observation['geese']) - get_n_geese_alive(current_observation['geese'])


def _get_goose_growth(current_observation, previous_observation, configuration):
    """ Returns 1 if the goose is bigger, 0 otherwise """
    current_len = _get_geese_len(current_observation)[current_observation['index']]
    previous_len = _get_geese_len(previous_observation)[previous_observation['index']]
    if current_observation['step'] % configuration['hunger_rate'] == 0 and current_observation['step']:
        current_len += 1
    if current_len > previous_len:
        return 1
    return 0


def _get_terminal_kill_and_grow_reward_params_from_name(reward_name):
    """ terminal_kill_and_grow_reward_10_2_1 """
    terminal_reward_scale, kill_reward, grow_reward = [float(value) for value in reward_name.split('_')[-3:]]
    return terminal_reward_scale, kill_reward, grow_reward


def get_death_reward_from_name(reward_name):
    if reward_name == 'sparse_reward':
        raise NotImplementedError()
    elif reward_name.startswith('ranking_reward'):
        raise NotImplementedError()
    elif reward_name.startswith('clipped_len_reward'):
        return _get_clipped_len_reward_params_from_name(reward_name)[0]
    elif reward_name.startswith('grow_and_kill_reward'):
        return _get_grow_and_kill_reward_params_from_name(reward_name)[0]
    elif reward_name.startswith('just_survive'):
        return float(reward_name.split('_')[-1])
    else:
        raise KeyError(reward_name)
