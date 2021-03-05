import pytest

from hungry_geese.state import (
    get_steps_to_shrink, get_steps_to_die, get_steps_to_end, get_n_geese_alive,
    get_reward
)

@pytest.mark.parametrize('step, hunger_rate, steps_to_shrink',  [
    (0, 40, 40),
    (1, 40, 39),
    (39, 40, 1),
    (40, 40, 40),
    (0, 20, 20),
])
def test_steps_to_shrink(step, hunger_rate, steps_to_shrink):
    assert steps_to_shrink == get_steps_to_shrink(step, hunger_rate)

@pytest.mark.parametrize('step, hunger_rate, goose_len, steps_to_die',  [
    (0, 40, 1, 40),
    (1, 40, 1, 39),
    (0, 40, 2, 80),
    (1, 40, 2, 79),
    (39, 40, 1, 1),
    (39, 40, 2, 41),
    (39, 40, 3, 81),
    (19, 20, 1, 1),
    (19, 20, 0, 0),
])
def test_steps_to_die(step, hunger_rate, goose_len, steps_to_die):
    assert steps_to_die == get_steps_to_die(step, hunger_rate, goose_len)

@pytest.mark.parametrize('step, episode_steps, steps_to_end',  [
    (0, 200, 200),
    (1, 200, 199),
])
def test_steps_to_end(step, episode_steps, steps_to_end):
    assert steps_to_end == get_steps_to_end(step, episode_steps)

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
