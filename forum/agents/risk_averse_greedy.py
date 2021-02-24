"""
https://www.kaggle.com/leontkh/hard-coded-passive-aggressive-agent
"""


from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import numpy as np

def get_nearest_cells(x,y):
    # returns all cells reachable from the current one
    result = []
    for i in (-1,+1):
        result.append(((x+i+7)%7, y))
        result.append((x, (y+i+11)%11))
    return result

def find_closest_food(table):
    # returns the first step toward the closest food item
    new_table = table.copy()

    updated = False
    for roll, axis, code in [
        (1, 0, 1),
        (-1, 0, 2),
        (1, 1, 3),
        (-1, 1, 4)
    ]:

        shifted_table = np.roll(table, roll, axis)

        if (table == -2).any() and (shifted_table[table == -2] == -3).any(): # we have found some food at the first step
            return code
        else:
            mask = np.logical_and(new_table == 0,shifted_table == -3)
            if mask.sum() > 0:
                updated = True
            new_table += code * mask
        if (table == -2).any() and shifted_table[table == -2][0] > 0: # we have found some food
            return shifted_table[table == -2][0]

        # else - update new reachible cells
        mask = np.logical_and(new_table == 0,shifted_table > 0)
        if mask.sum() > 0:
            updated = True
        new_table += shifted_table * mask

    # if we updated anything - continue reccurison
    if updated:
        return find_closest_food(new_table)
    # if not - return some step
    else:
        return table.max()

def agent(obs_dict, config_dict):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)


    table = np.zeros((7,11))
    # 0 - emply cells
    # -1 - obstacles
    # -2 - food
    # -3 - head
    # 1,2,3,4 - reachable on the current step cell, number is the id of the first step direction

    legend = {
        1: 'SOUTH',
        2: 'NORTH',
        3: 'EAST',
        4: 'WEST'
    }

    # let's add food to the map
    for food in observation.food:
        x,y = row_col(food, configuration.columns)
        table[x,y] = -2 # food

    # let's add all cells that are forbidden
    for i in range(4):
        opp_goose = observation.geese[i]
        if len(opp_goose) == 0:
            continue
        for n in opp_goose[:-1]:
            x,y = row_col(n, configuration.columns)
            table[x,y] = -1 # forbidden cells
        if i != player_index:
            x,y = row_col(opp_goose[0], configuration.columns)
            possible_moves = get_nearest_cells(x,y) # head can move anywhere
            for x,y in possible_moves:
                table[x,y] = -1 # forbidden cells


    # let's add head position
    x,y = row_col(player_head, configuration.columns)
    table[x,y] = -3

    # the first step toward the nearest food
    step = int(find_closest_food(table))

    # if there is not available steps make random step
    if step not in [1,2,3,4]:
        step = np.random.randint(4) + 1

    return legend[step]