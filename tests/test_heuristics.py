import pytest

from hungry_geese.heuristic import is_future_position_doomed, is_food_around_head

@pytest.mark.parametrize('future_position, observation, is_doomed', [
    (1, {'index': 0, 'geese': [[1, 2]]}, 1),
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