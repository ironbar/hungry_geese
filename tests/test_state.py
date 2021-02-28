import pytest

from hungry_geese.state import get_steps_to_shrink, get_steps_to_die, get_steps_to_end

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
])
def test_steps_to_die(step, hunger_rate, goose_len, steps_to_die):
    assert steps_to_die == get_steps_to_die(step, hunger_rate, goose_len)

@pytest.mark.parametrize('step, episode_steps, steps_to_end',  [
    (0, 200, 200),
    (1, 200, 199),
])
def test_steps_to_end(step, episode_steps, steps_to_end):
    assert steps_to_end == get_steps_to_end(step, episode_steps)