import pytest

from hungry_geese.reward import (
    get_n_geese_alive, get_sparse_reward, get_ranking_reward
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