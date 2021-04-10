import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions

def get_certain_death_mask(observation, configuration):
    """
    Returns a binary mask for the actions that has ones on those actions that lead to a certain
    death

    array([ [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
            [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]])
    """
    geese = observation['geese']
    head_position = geese[observation['index']][0]
    future_positions = adjacent_positions(head_position, configuration['columns'], configuration['rows'])
    certain_death_mask = np.array([is_future_position_doomed(future_position, observation, configuration) for future_position in future_positions], dtype=np.float32)
    return certain_death_mask

def is_future_position_doomed(future_position, observation, configuration):
    """
    Returns
    -------
    0 if future position is safe
    1 if death is certain
    0.5 if death depends on other goose movements
    """
    is_doomed = 0
    for idx, goose in enumerate(observation['geese']):
        if not goose:
            continue
        # If the future position is on the body of the goose, then death is certain
        if future_position in goose[:-1]:
            return 1
        # The tail is special, if the goose is not going to grow it is safe
        if future_position == goose[-1] and idx != observation['index']:
            if is_food_around_head(goose, observation['food'], configuration):
                is_doomed = 0.5
        # Check if other goose may move into that position
        if idx != observation['index']:
            if is_food_around_head(goose, [future_position], configuration):
                is_doomed = 0.5
    return is_doomed

def is_food_around_head(goose, food, configuration):
    future_positions = adjacent_positions(goose[0], configuration['columns'], configuration['rows'])
    return any(food_position in future_positions for food_position in food)
