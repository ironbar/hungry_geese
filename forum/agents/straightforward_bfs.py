"""
https://www.kaggle.com/ihelon/hungry-geese-agents-comparison
"""

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import numpy as np



def bfs(start_x, start_y, mask, food_coords):
    dist_matrix = np.zeros_like(mask)
    vect_matrix = np.full_like(mask, -1)

    queue = [(start_x, start_y, 0, None)]

    while queue:
        current_x, current_y, current_dist, vect = queue.pop(0)
        vect_matrix[current_x, current_y] = vect
        up_x = current_x + 1 if current_x != 6 else 0
        down_x = current_x - 1 if current_x != 0 else 6
        left_y = current_y - 1 if current_y != 0 else 10
        right_y = current_y + 1 if current_y != 10 else 0

        if mask[up_x, current_y] != -1 and not dist_matrix[up_x, current_y]:
            dist_matrix[up_x, current_y] = current_dist + 1
            if vect is None:
                queue.append((up_x, current_y, current_dist + 1, 0))
            else:
                queue.append((up_x, current_y, current_dist + 1, vect))
        if mask[down_x, current_y] != -1 and not dist_matrix[down_x, current_y]:
            dist_matrix[down_x, current_y] = current_dist + 1
            if vect is None:
                queue.append((down_x, current_y, current_dist + 1, 1))
            else:
                queue.append((down_x, current_y, current_dist + 1, vect))
        if mask[current_x, left_y] != -1 and not dist_matrix[current_x, left_y]:
            dist_matrix[current_x, left_y] = current_dist + 1
            if vect is None:
                queue.append((current_x, left_y, current_dist + 1, 2))
            else:
                queue.append((current_x, left_y, current_dist + 1, vect))
        if mask[current_x, right_y] != -1 and not dist_matrix[current_x, right_y]:
            dist_matrix[current_x, right_y] = current_dist + 1
            if vect is None:
                queue.append((current_x, right_y, current_dist + 1, 3))
            else:
                queue.append((current_x, right_y, current_dist + 1, vect))

    min_food_id = -1
    min_food_dist = np.inf
    for id_, food in enumerate(food_coords):
        if dist_matrix[food[0], food[1]] != 0 and min_food_dist > dist_matrix[food[0], food[1]]:
            min_food_id = id_
            min_food_dist = dist_matrix[food[0], food[1]]

    if min_food_id == -1:
        x, y = -1, -1
        mn = 0
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                if dist_matrix[i, j] > mn:
                    x, y = i, j
                    mn = dist_matrix[i, j]
        return vect_matrix[x, y]

    food_x, food_y = food_coords[min_food_id]
    return vect_matrix[food_x, food_y]


LAST_ACTION = None


def straightforward_bfs(obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index

    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    start_row, start_col = row_col(player_head, configuration.columns)

    mask = np.zeros((configuration.rows, configuration.columns))
    for current_id in range(4):
        current_goose = observation.geese[current_id]
        for block in current_goose:
            current_row, current_col = row_col(block, configuration.columns)
            mask[current_row, current_col] = -1

    food_coords = []

    for food_id in range(configuration.min_food):
        food = observation.food[food_id]
        current_row, current_col = row_col(food, configuration.columns)
        mask[current_row, current_col] = 2
        food_coords.append((current_row, current_col))


    last_action = bfs(start_row, start_col, mask, food_coords)

    global LAST_ACTION
    up_x = start_row + 1 if start_row != 6 else 0
    down_x = start_row - 1 if start_row != 0 else 6
    left_y = start_col - 1 if start_col != 0 else 10
    right_y = start_col + 1 if start_col != 10 else 0

    step = Action.NORTH.name
    if last_action == 0:
        step = Action.SOUTH.name
        if LAST_ACTION == Action.NORTH.name:
            if mask[down_x, start_col] != -1:
                step = Action.NORTH.name
            elif mask[start_row, left_y] != -1:
                step = Action.WEST.name
            elif mask[start_row, right_y] != -1:
                step = Action.EAST.name
    if last_action == 1:
        step = Action.NORTH.name
        if LAST_ACTION == Action.SOUTH.name:
            if mask[up_x, start_col] != -1:
                step = Action.SOUTH.name
            elif mask[start_row, left_y] != -1:
                step = Action.WEST.name
            elif mask[start_row, right_y] != -1:
                step = Action.EAST.name
    if last_action == 2:
        step = Action.WEST.name
        if LAST_ACTION == Action.EAST.name:
            if mask[up_x, start_col] != -1:
                step = Action.SOUTH.name
            elif mask[down_x, start_col] != -1:
                step = Action.NORTH.name
            elif mask[start_row, right_y] != -1:
                step = Action.EAST.name
    if last_action == 3:
        step = Action.EAST.name
        if LAST_ACTION == Action.WEST.name:
            if mask[up_x, start_col] != -1:
                step = Action.SOUTH.name
            elif mask[down_x, start_col] != -1:
                step = Action.NORTH.name
            elif mask[start_row, left_y] != -1:
                step = Action.WEST.name
    LAST_ACTION = step

    return step