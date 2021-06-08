import pytest
import numpy as np

from hungry_geese.reward import (
    get_n_geese_alive, get_sparse_reward, get_ranking_reward, get_cumulative_reward,
    get_clipped_len_reward, get_grow_and_kill_reward, get_just_survive_reward,
    get_death_reward_from_name, get_terminal_kill_and_grow_reward,
    _get_terminal_sparse_reward, get_reward
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
    (np.array([1., 0, 2]), 'grow_and_kill_reward_-1_3_3_1', np.array([1, 1, 2])),
    (np.array([1., 0, 2]), 'grow_and_kill_reward_-1_3_3_1', np.array([1, 1, 2])),
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

@pytest.mark.parametrize("current_observation, previous_observation, reward_name, reward",[
    ({'geese': [[1], [2]], 'index':0}, {'geese': [[1], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 0),
    ({'geese': [[1, 2], [2]], 'index':0}, {'geese': [[1], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 1), #grow
    ({'geese': [[], [2]], 'index':0}, {'geese': [[1], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', -1), #die
    ({'geese': [[], [2]], 'index':1}, {'geese': [[1], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 1), #kill
    ({'geese': [[], [2]], 'index':1}, {'geese': [[1], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_2', 2), #kill
    ({'geese': [[1, 2, 3], [2]], 'index':0}, {'geese': [[1, 2], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 1), #grow
    ({'geese': [[1, 2, 3, 4], [2]], 'index':0}, {'geese': [[1, 2, 3], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 1), #grow
    ({'geese': [[1, 2, 3, 4, 5], [2]], 'index':0}, {'geese': [[1, 2, 3, 4], [2]], 'index':0}, 'grow_and_kill_reward_-1_8_3_1', 0), #do not grow too much
])
def test_grow_and_kill_reward(current_observation, previous_observation, reward_name, reward):
    assert reward == get_grow_and_kill_reward(current_observation, previous_observation, reward_name)

@pytest.mark.parametrize("current_observation, reward_name, reward",[
    ({'geese': [[1], [2]], 'index':0}, 'just_survive_-1', 0),
    ({'geese': [[], [2]], 'index':0}, 'just_survive_-1', -1),
    ({'geese': [[], [2]], 'index':1}, 'just_survive_-1', 0),
    ({'geese': [[], [2]], 'index':0}, 'just_survive_-5', -5),
])
def test_just_survive_reward(current_observation, reward_name, reward):
    assert reward == get_just_survive_reward(current_observation, reward_name)


@pytest.mark.parametrize('reward_name, death_reward', [
    ('just_survive_-5', -5),
    ('grow_and_kill_reward_-1_8_3_2', -1),
])
def test_get_death_reward_from_name(reward_name, death_reward):
    assert death_reward == get_death_reward_from_name(reward_name)

@pytest.mark.parametrize('reward_name', ['terminal_kill_and_grow_10_2_1'])
@pytest.mark.parametrize('configuration', [{'episodeSteps': 200, 'hunger_rate': 40}])
@pytest.mark.parametrize('reward, current_observation, previous_observation', [
    (0, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # no change
    (2, {'geese': [[1], [], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # kill other
    (-25, {'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die on last position
    (1, {'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow
    (3, {'geese': [[1, 2], [], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow and kill
    (5, {'geese': [[1, 2], [], [], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow and kill 2
    (3, {'geese': [[1, 2], [], [], [4]], 'index':0, 'step': 5}, {'geese': [[1], [], [3], [4]], 'index':0}), # grow and kill
    (0, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1, 2], [2], [3], [4]], 'index':0}), # decrease
    (-10, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 199}, {'geese': [[1], [2], [3], [4]], 'index':0}), # end on tie
    (5, {'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 199}, {'geese': [[1], [2], [3], [4]], 'index':0}), # win
    (5, {'geese': [[1, 2], [], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # win
    (1, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 40}, {'geese': [[1], [2], [3], [4]], 'index':0}), # eat when geese decreases
])
def test_get_terminal_kill_and_grow_reward(reward, current_observation, previous_observation, reward_name, configuration):
    assert reward == get_terminal_kill_and_grow_reward(current_observation, previous_observation, reward_name, configuration)

@pytest.mark.parametrize('reward, current_observation, previous_observation', [
    (0, {'geese': [[], [2], [3], [4]], 'index':0}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die on last position
    (0.5, {'geese': [[], [], [3], [4]], 'index':0}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die at same time
    (1, {'geese': [[], [], [], [4]], 'index':0}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die at same time
    (1.5, {'geese': [[], [], [], []], 'index':0}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die at same time
    (2, {'geese': [[], [], [], []], 'index':0}, {'geese': [[1], [2], [3], []], 'index':0}), # die at same time
    (2.5, {'geese': [[], [], [], []], 'index':0}, {'geese': [[1], [2], [], []], 'index':0}), # die at same time
    (3, {'geese': [[1], [], [], []], 'index':0}, {'geese': [[1], [2], [], []], 'index':0}), # wins
    (3, {'geese': [[1, 2], [], [], []], 'index':0}, {'geese': [[1], [2], [], []], 'index':0}), # wins
    (3, {'geese': [[1, 2], [2], [], []], 'index':0}, {'geese': [[1], [2], [], []], 'index':0}), # wins
])
def test_get_terminal_sparse_reward(reward, current_observation, previous_observation):
    assert reward == _get_terminal_sparse_reward(current_observation, previous_observation)


@pytest.mark.parametrize('reward_name', ['v2_terminal_kill_and_grow_reward_2_-5_5_2_1'])
@pytest.mark.parametrize('configuration', [{'episodeSteps': 200, 'hunger_rate': 40}])
@pytest.mark.parametrize('reward, current_observation, previous_observation', [
    (0, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # no change
    (2, {'geese': [[1], [], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # kill other
    (-10, {'geese': [[], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # die on last position
    (-8, {'geese': [[], [2], [3], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], []], 'index':0}), # die on 3 position
    (-6, {'geese': [[], [2], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [], []], 'index':0}), # die on 2 position
    (-5, {'geese': [[], [], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [], []], 'index':0}), # tie on 2 position
    (1, {'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow
    (3, {'geese': [[1, 2], [], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow and kill
    (5, {'geese': [[1, 2], [], [], [4]], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # grow and kill 2
    (3, {'geese': [[1, 2], [], [], [4]], 'index':0, 'step': 5}, {'geese': [[1], [], [3], [4]], 'index':0}), # grow and kill
    (0, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 5}, {'geese': [[1, 2], [2], [3], [4]], 'index':0}), # decrease
    (-7, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 199}, {'geese': [[1], [2], [3], [4]], 'index':0}), # end on tie
    (5, {'geese': [[1, 2], [2], [3], [4]], 'index':0, 'step': 199}, {'geese': [[1], [2], [3], [4]], 'index':0}), # win
    (5, {'geese': [[1, 2], [], [], []], 'index':0, 'step': 5}, {'geese': [[1], [2], [3], [4]], 'index':0}), # win
    (1, {'geese': [[1], [2], [3], [4]], 'index':0, 'step': 40}, {'geese': [[1], [2], [3], [4]], 'index':0}), # eat when geese decreases
])
def test_get_v2_terminal_kill_and_grow_reward(reward, current_observation, previous_observation, reward_name, configuration):
    assert reward == get_reward(current_observation, previous_observation, configuration, reward_name)