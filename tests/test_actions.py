import pytest

from hungry_geese.actions import get_action_from_relative_movement

@pytest.mark.parametrize('relative_movement, previous_action, action', [
    (1, 'NORTH', 'NORTH'),
    (1, 'EAST', 'EAST'),
    (1, 'SOUTH', 'SOUTH'),
    (1, 'WEST', 'WEST'),
    (2, 'NORTH', 'EAST'),
    (0, 'NORTH', 'WEST'),
    (0, 'SOUTH', 'EAST'),
    (2, 'SOUTH', 'WEST'),
])
def test_get_action_from_relative_movement(relative_movement, previous_action, action):
    assert action == get_action_from_relative_movement(relative_movement, previous_action)
