import pytest
import tensorflow as tf

from hungry_geese.utils import (
    random_legal_action, get_timestamp,
    log_to_tensorboard, log_configuration_to_tensorboard)

@pytest.mark.parametrize('previous_action, ilegal_next_action', [
    ('NORTH', 'SOUTH'),
    ('SOUTH', 'NORTH'),
    ('EAST', 'WEST'),
    ('WEST', 'EAST'),
])
def test_random_legal_action(previous_action, ilegal_next_action):
    assert ilegal_next_action not in [random_legal_action(previous_action) for _ in range(10)]

def test_get_timestamp_is_callable():
    get_timestamp()

def test_log_to_tensorboard(tmpdir):
    tensorboard_writer = tf.summary.create_file_writer(str(tmpdir))
    log_to_tensorboard('key', 0, 0, tensorboard_writer)

def test_log_configuration_to_tensorboard(tmpdir):
    tensorboard_writer = tf.summary.create_file_writer(str(tmpdir))
    log_configuration_to_tensorboard(dict(key=0), tensorboard_writer)
