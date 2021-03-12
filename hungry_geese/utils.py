"""
Functions commonly used in the challenge
"""
import os
import psutil
import time
import random
import tensorflow as tf

from hungry_geese.definitions import ACTIONS

def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp

def opposite_action(action):
    action_to_opposite = {
        'NORTH': 'SOUTH',
        'EAST': 'WEST',
        'SOUTH': 'NORTH',
        'WEST': 'EAST',
    }
    return action_to_opposite[action]

def random_legal_action(previous_action):
    """ Returns a random action that is legal """
    if previous_action is not None:
        opposite = opposite_action(previous_action)
        options = [action for action in ACTIONS if action != opposite]
    else:
        options = ACTIONS
    action = random.choice(options)
    return action

def log_to_tensorboard(key, value, epoch, tensorboard_writer):
    with tensorboard_writer.as_default():
        tf.summary.scalar(key, value, step=epoch)

def log_configuration_to_tensorboard(configuration, tensorboard_writer, step=0):
    with tensorboard_writer.as_default():
        for key, value in configuration.items():
            tf.summary.text(key, str(value), step=step)

def get_ram_usage_and_available():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss/1e9
    stats = psutil.virtual_memory()  # returns a named tuple
    ram_available = getattr(stats, 'available')/1e9
    return ram_usage, ram_available


def get_cpu_usage():
    return psutil.cpu_percent()
