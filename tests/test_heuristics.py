import pytest
import numpy as np

from hungry_geese.heuristic import (
    is_future_position_doomed, is_food_around_head,
    get_certain_death_mask, adapt_mask_to_3d_action
)


@pytest.mark.parametrize('observation, mask', [
    ({'index': 0, 'geese': [[1, 2]]}, np.zeros(4)),
    ({'index': 0, 'geese': [[1, 2, 13, 12, 11]]}, np.array([0, 1, 1, 0])),
    ({'index': 0, 'geese': [[1, 2, 13, 12]]}, np.array([0, 1, 0, 0])),
    ({'index': 0, 'geese': [[1, 2, 13, 12], [55, 66, 0, 11]]}, np.array([0, 1, 0, 1])),
    ({'index': 0, 'geese': [[1, 2, 13, 12], [66, 0, 11]]}, np.array([0.5, 1, 0, 1])),
    ({'index': 0, 'geese': [[12, 1, 2, 13, 24, 23, 22, 11, 0]]}, np.array([1, 1, 1, 1])),
])
def test_get_certain_death_mask(observation, mask):
    configuration = dict(columns=11, rows=7)
    certain_death_mask = get_certain_death_mask(observation, configuration)
    assert pytest.approx(mask) == certain_death_mask


@pytest.mark.parametrize('future_position, observation, is_doomed', [
    (1, {'index': 0, 'geese': [[1, 2]]}, 1),
    (1, {'index': 0, 'geese': [[1]]}, 0), # if the len is 1 it is safe to move to the head
    (63, {'index': 0, 'geese': [[1, 2]]}, 0),
    (63, {'index': 0, 'geese': [[1, 2], []]}, 0),
    (2, {'index': 0, 'geese': [[], [1, 2]], 'food': [0]}, 0.5),
    (2, {'index': 0, 'geese': [[], [0, 1, 2]], 'food': [63]}, 0),
    (2, {'index': 1, 'geese': [[], [1, 2]], 'food': [0]}, 0),
    (2, {'index': 0, 'geese': [[], [1]]}, 0.5),
])
def test_is_future_position_doomed(future_position, observation, is_doomed):
    configuration = dict(columns=11, rows=7)
    assert is_doomed == is_future_position_doomed(future_position, observation, configuration)

@pytest.mark.parametrize('goose, food, is_food', [
    ([12], [], False),
    ([12], [13], True),
    ([12], [11], True),
    ([12], [1], True),
    ([12], [23], True),
    ([12], [24], False),
    ([12], [23, 24], True),
])
def test_is_food_around_head(goose, food, is_food):
    configuration = dict(columns=11, rows=7)
    assert is_food == is_food_around_head(goose, food, configuration)

@pytest.mark.parametrize('mask_4d, previous_action, mask_3d', [
    ([1, 2, 3, 4], 'NORTH', [4, 1, 2]),
    ([1, 2, 3, 4], 'EAST', [1, 2, 3]),
    ([1, 2, 3, 4], 'SOUTH', [2, 3, 4]),
    ([1, 2, 3, 4], 'WEST', [3, 4, 1]),
])
def test_adapt_mask_to_3d_action(mask_4d, previous_action, mask_3d):
    assert mask_3d == adapt_mask_to_3d_action(np.array(mask_4d), previous_action).tolist()
