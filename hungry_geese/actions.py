"""
Actions
"""
from hungry_geese.definitions import ACTIONS, ACTION_TO_IDX

def get_action_from_relative_movement(relative_movement, previous_action):
    """
    Returns the action using the previous action and relative_movement(0,1,2s)
    """
    return ACTIONS[(relative_movement - 1 + ACTION_TO_IDX[previous_action])%len(ACTIONS)]
