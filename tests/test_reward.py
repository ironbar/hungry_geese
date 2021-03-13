import pytest

from hungry_geese.reward import get_n_geese_alive, get_reward

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
def test_reward(current_observation, previous_observation, configuration, reward):
    assert reward == get_reward(current_observation, previous_observation, configuration)