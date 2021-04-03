import pytest
import numpy as np
from kaggle_environments import make

from hungry_geese.state import (
    get_steps_to_shrink, get_steps_to_die, get_steps_to_end,
    GameState, horizontal_simmetry, vertical_simmetry, player_simmetry,
    permutations, combine_data, apply_all_simetries, get_ohe_opposite_actions
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

@pytest.fixture
def train_data():
    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy", "greedy", "greedy"])
    configuration = env.configuration
    state = GameState()

    obs = trainer.reset()
    done = False
    action = 'NORTH'
    while not done:
        state.update(obs, configuration)
        state.add_action(action)
        obs, reward, done, info = trainer.step(action)
    state.update(obs, configuration)
    return state.prepare_data_for_training()

def test_horizontal_simmetry_is_invertible(train_data):
    inverted_data = horizontal_simmetry(horizontal_simmetry(train_data))
    assert all((train_data[idx] == inverted_data[idx]).all() for idx in range(4))

def test_vertical_simmetry_is_invertible(train_data):
    inverted_data = vertical_simmetry(vertical_simmetry(train_data))
    assert all((train_data[idx] == inverted_data[idx]).all() for idx in range(4))

@pytest.mark.parametrize('new_positions', list(permutations([0, 1, 2])))
def test_player_simmetry_is_invertible(train_data, new_positions):
    inverted_data = player_simmetry(player_simmetry(train_data, [2, 1, 0]), [2, 1, 0])
    assert all((train_data[idx] == inverted_data[idx]).all() for idx in range(4))

def test_combine_data(train_data):
    new_data = combine_data([train_data, train_data])
    assert len(new_data[0]) == len(train_data[0])*2

def test_apply_all_simetries(train_data):
    new_data = apply_all_simetries(train_data)
    assert len(new_data[0]) == len(train_data[0])*24

def test_get_ohe_opposite_actions():
    actions = np.eye(4)
    opposite_actions = get_ohe_opposite_actions(actions)
    assert pytest.approx(actions[:2]) == opposite_actions[2:]
    assert pytest.approx(actions) == get_ohe_opposite_actions(opposite_actions)