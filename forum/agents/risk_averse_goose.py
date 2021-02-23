"""
https://www.kaggle.com/ilialar/risk-averse-greedy-goose
"""
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import numpy as np
import random

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


    # (direction of the step, axis, code)
    possible_moves = [
        (1, 0, 1),
        (-1, 0, 2),
        (1, 1, 3),
        (-1, 1, 4)
    ]

    # shuffle possible options to add variability
    random.shuffle(possible_moves)


    updated = False
    for roll, axis, code in possible_moves:

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

last_step = None

def agent(obs_dict, config_dict):
    global last_step

    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)


    table = np.zeros((7,11))
    # 0 - emply cells
    # -1 - obstacles
    # -4 - possible obstacles
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

        is_close_to_food = False

        if i != player_index:
            x,y = row_col(opp_goose[0], configuration.columns)
            possible_moves = get_nearest_cells(x,y) # head can move anywhere

            for x,y in possible_moves:
                if table[x,y] == -2:
                    is_close_to_food = True

                table[x,y] = -4 # possibly forbidden cells

        # usually we ignore the last tail cell but there are exceptions
        tail_change = -1
        if obs_dict['step'] % 40 == 39:
            tail_change -= 1

        # we assume that the goose will eat the food
        if is_close_to_food:
            tail_change += 1
        if tail_change >= 0:
            tail_change = None


        for n in opp_goose[:tail_change]:
            x,y = row_col(n, configuration.columns)
            table[x,y] = -1 # forbidden cells

    # going back is forbidden according to the new rules
    x,y = row_col(player_head, configuration.columns)
    if last_step is not None:
        if last_step == 1:
            table[(x + 6) % 7,y] = -1
        elif last_step == 2:
            table[(x + 8) % 7,y] = -1
        elif last_step == 3:
            table[x,(y + 10)%11] = -1
        elif last_step == 4:
            table[x,(y + 12)%11] = -1

    # add head position
    table[x,y] = -3

    # the first step toward the nearest food
    step = int(find_closest_food(table))

    # if there is not available steps try to go to possibly dangerous cell
    if step not in [1,2,3,4]:
        x,y = row_col(player_head, configuration.columns)
        if table[(x + 8) % 7,y] == -4:
            step = 1
        elif table[(x + 6) % 7,y] == -4:
            step = 2
        elif table[x,(y + 12)%11] == -4:
            step = 3
        elif table[x,(y + 10)%11] == -4:
            step = 4

    # else - do a random step and lose
        else:
            step = np.random.randint(4) + 1

    last_step = step
    return legend[step]