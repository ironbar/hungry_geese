"""
Functions commonly used in the challenge
"""
import time
import random

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
