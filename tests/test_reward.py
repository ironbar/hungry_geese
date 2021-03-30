import pytest
import numpy as np

from hungry_geese.reward import (
    get_n_geese_alive, get_sparse_reward, get_ranking_reward, get_cumulative_reward,
    get_clipped_len_reward
)

@pytest.mark.parametrize('geese, n',  [
    ([[1], [1]], 2),
    ([[1], []], 1),
])
def test_get_n_geese_alive(geese, n):
    assert n == get_n_geese_alive(geese)

@pytest.mark.parametrize("current_observation, previous_observation, configuration, reward",[
    ({'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 200}, 0),
    ({'geese': [[1], [2], [3], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 200}, 1),
    ({'geese': [[1], [2], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 200}, 2),
    ({'geese': [[1], [], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 200}, 3),
    ({'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 200}, -1),
    ({'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, 1.5),
    ({'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, 3),
    ({'geese': [[1, 2], [2, 3], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, 2.5),
    ({'geese': [[1], [2], [3], [4, 5]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, 1),
    ({'geese': [[1], [2, 3], [3, 4], [4, 5]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, 0),
    ({'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]]}, {'episodeSteps': 6}, -1),
])
def test_sparse_reward(current_observation, previous_observation, configuration, reward):
    assert reward == get_sparse_reward(current_observation, previous_observation, configuration)

@pytest.mark.parametrize("current_observation, reward_name, reward",[
    ({'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, 'ranking_reward', 1.5),
    ({'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 5}, 'ranking_reward', 3),
    ({'geese': [[1, 2], [2, 3], [3, 4], [4]], 'index':3, 'step': 5}, 'ranking_reward', 0),
    ({'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, 'ranking_reward_-1_2', -1),
    ({'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, 'ranking_reward_-10_2', -10),
])
def test_ranking_reward(current_observation, reward_name, reward):
    assert reward == get_ranking_reward(current_observation, reward_name)

@pytest.mark.parametrize("rewards, reward_name, cumulative_reward",[
    (np.ones(5), 'ranking_reward_-1_2', np.ones(5)),
    (np.ones(5), 'ranking_reward_-1_3', np.ones(5)),
    (np.ones(5), 'ranking_reward_-1_1', np.ones(5)),
    (np.ones(5), 'ranking_reward_-1_4', np.ones(5)),
    (np.array([1., 1, 3]), 'ranking_reward_-1_3', np.array([5/3, 2, 3])),
    (np.array([1., 0, 2]), 'ranking_reward_-1_3', np.array([1, 1, 2])),
    (np.ones(1), 'ranking_reward_-1_1', np.ones(1)),
    (np.ones(1), 'ranking_reward_-1_2', np.ones(1)),
    (np.array([1., 0, 2]), 'clipped_len_reward_-10_3_2_-5', np.array([1, 1, 2])),
    (np.array([1., 0, 2]), 'clipped_len_reward_-10_3_2_-5', np.array([1, 1, 2])),
])
def test_get_cumulative_reward(rewards, reward_name, cumulative_reward):
    assert pytest.approx(cumulative_reward) == get_cumulative_reward(rewards, reward_name)

@pytest.mark.parametrize("current_observation, reward_name, reward",[
    ({'geese': [[1], [2]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 0),
    ({'geese': [[1], [2]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 0),
    ({'geese': [[1, 2], [2]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 1),
    ({'geese': [[1, 2, 3], [2]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 2),
    ({'geese': [[1, 2, 3, 4], [2]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 2),
    ({'geese': [[1, 2, 3, 4], [2]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_3_-5', 3),
    ({'geese': [[1, 2], [2]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -1),
    ({'geese': [[1, 2, 3], [2]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -2),
    ({'geese': [[1, 2, 3, 4], [2]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -3),
    ({'geese': [[1, 2, 3, 4], [2]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_3_-2', -2),
    ({'geese': [[1, 2, 3, 4], []], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_3_-2', -10),
    ({'geese': [[1], [2], [3]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 0),
    ({'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 0),
    ({'geese': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4]], 'index':0, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', 1),
    ({'geese': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4]], 'index':1, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -1),
    ({'geese': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4]], 'index':2, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -2),
    ({'geese': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4]], 'index':3, 'step': 5}, 'clipped_len_reward_-10_4_2_-5', -3),
    ({'geese': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4]], 'index':3, 'step': 5}, 'clipped_len_reward_-10_4_2_-2', -2),
])
def test_clipped_len_reward(current_observation, reward_name, reward):
    assert reward == get_clipped_len_reward(current_observation, reward_name)