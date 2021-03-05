import pytest

from hungry_geese.utils import random_legal_action

@pytest.mark.parametrize('previous_action, ilegal_next_action', [
    ('NORTH', 'SOUTH'),
    ('SOUTH', 'NORTH'),
    ('EAST', 'WEST'),
    ('WEST', 'EAST'),
])
def test_random_legal_action(previous_action, ilegal_next_action):
    assert ilegal_next_action not in [random_legal_action(previous_action) for _ in range(10)]