"""
https://www.kaggle.com/ihelon/hungry-geese-agents-comparison
"""

# Base code for this from
# https://www.kaggle.com/ilialar/risk-averse-greedy-goose

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

# Moves constants
SOUTH = 1
NORTH = 2
EAST  = 3
WEST  = 4
REVERSE_MOVE = {
    None : None,
    SOUTH: NORTH,
    NORTH: SOUTH,
    EAST : WEST,
    WEST : EAST,
}
CIRCLE_MOVE = {
    None : None,
    SOUTH: WEST,
    NORTH: EAST,
    EAST : SOUTH,
    WEST : NORTH
}

# Board constants
MY_HEAD             =  2
FOOD_CELL           =  1
EMPTY               =  0
HEAD_POSSIBLE_CELL  = -1
BODY_CELL           = -2

# Store last move
last_move = None
last_eaten = 0
last_size = 1
step = 0

# Returns a list of possible destinations in order to reach `dest_cell`
def move_towards (head_cell, neck_cell, dest_cell, configuration):
    print ("--- Computing food movements...")
    destinations = []
    x_head, y_head = row_col(head_cell, configuration.columns)
    x_neck, y_neck = row_col(neck_cell, configuration.columns)
    x_dest, y_dest = row_col(dest_cell, configuration.columns)
    print ("-> Head at ({}, {})".format(x_head, y_head))
    print ("-> Neck at ({}, {})".format(x_neck, y_neck))
    print ("-> Dest at ({}, {})".format(x_dest, y_dest))
    dx = x_head - x_dest
    dy = y_head - y_dest
    if (dx >= 4):
        dx = 7 - dx
    elif (dx <= -4):
        dx += 7
    if (dy >= 6):
        dy = 11 - dy
    elif (dy <= -6):
        dy += 11
    print ("dx={}, dy={}".format(dx, dy))
    if (dx > 0):
        x_move = (x_head - 1 + 7) % 7
        y_move = y_head
        print ("Move ({}, {}), Neck ({}, {})".format(x_move, y_move, x_neck, y_neck))
        if not ((x_move == x_neck) and (y_move == y_neck)):
            destinations.append((x_move, y_move, NORTH))
    elif (dx < 0):
        x_move = (x_head + 1 + 7) % 7
        y_move = y_head
        print ("Move ({}, {}), Neck ({}, {})".format(x_move, y_move, x_neck, y_neck))
        if not ((x_move == x_neck) and (y_move == y_neck)):
            destinations.append((x_move, y_move, SOUTH))
    if (dy > 0):
        x_move = x_head
        y_move = (y_head - 1 + 11) % 11
        print ("Move ({}, {}), Neck ({}, {})".format(x_move, y_move, x_neck, y_neck))
        if not ((x_move == x_neck) and (y_move == y_neck)):
            destinations.append((x_move, y_move, WEST))
    elif (dy < 0):
        x_move = x_head
        y_move = (y_head + 1 + 11) % 11
        print ("Move ({}, {}), Neck ({}, {})".format(x_move, y_move, x_neck, y_neck))
        if not ((x_move == x_neck) and (y_move == y_neck)):
            destinations.append((x_move, y_move, EAST))
    return destinations

def get_all_movements(goose_head, configuration):
    x_head, y_head = row_col(goose_head, configuration.columns)
    movements = []
    movements.append(((x_head - 1 + 7) % 7, y_head, NORTH))
    movements.append(((x_head + 1 + 7) % 7, y_head, SOUTH))
    movements.append((x_head, (y_head - 1 + 11) % 11, WEST))
    movements.append((x_head, (y_head + 1 + 11) % 11, EAST))
    return movements

def get_nearest_cells(x, y):
    # Returns adjacent cells from the current one
    result = []
    for i in (-1,+1):
        result.append(((x+i+7)%7, y))
        result.append((x, (y+i+11)%11))
    return result

# Compute L1 distance between cells
def cell_distance (a, b, configuration):
    xa, ya = row_col(a, configuration.columns)
    xb, yb = row_col(b, configuration.columns)
    dx = abs(xa - xb)
    dy = abs(ya - yb)
    if (dx >= 4):
        dx = 7 - dx
    if (dy >= 6):
        dy = 11 - dy
    return dx + dy

# Tells if that particular cell forbids movement on the next step
def is_closed (movement, board):
    return all([board[x_adj, y_adj] for (x_adj, y_adj) in get_nearest_cells(movement[0], movement[1])])

def is_safe (movement, board):
    return board[movement[0], movement[1]] >= 0

def is_half_safe (movement, board):
    return board[movement[0], movement[1]] >= -1

def agent (obs_dict, config_dict):
    global last_move
    global last_eaten
    global last_size
    global step
    print ("==============================================")
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)

    if (len(player_goose) > last_size):
        last_size = len(player_goose)
        last_eaten = step
    step += 1

    moves = {
        1: 'SOUTH',
        2: 'NORTH',
        3: 'EAST',
        4: 'WEST'
    }

    board = np.zeros((7, 11))

    # Adding food to board
    for food in observation.food:
        x, y = row_col(food, configuration.columns)
        print ("Food cell on ({}, {})".format(x, y))
        board[x, y] = FOOD_CELL

    # Adding geese to the board
    for i in range(4):
        goose = observation.geese[i]
        # Skip if goose is dead
        if len(goose) == 0:
            continue
        # If it's an opponent
        if i != player_index:
            x, y = row_col(goose[0], configuration.columns)
            # Add possible head movements for it
            for px, py in get_nearest_cells(x, y):
                print ("Head possible cell on ({}, {})".format(px, py))
                # If one of these head movements may lead the goose
                # to eat, add tail as BODY_CELL, because it won't move.
                if board[px, py] == FOOD_CELL:
                    x_tail, y_tail = row_col(goose[-1], configuration.columns)
                    print ("Adding tail on ({}, {}) as the goose may eat".format(x_tail, y_tail))
                    board[x_tail, y_tail] = BODY_CELL
                board[px, py] = HEAD_POSSIBLE_CELL
        # Adds goose body without tail (tail is previously added only if goose may eat)
        for n in goose[:-1]:
            x, y = row_col(n, configuration.columns)
            print ("Body cell on ({}, {})".format(x, y))
            board[x, y] = BODY_CELL

    # Adding my head to the board
    x, y = row_col(player_head, configuration.columns)
    print ("My head is at ({}, {})".format(x, y))
    board[x, y] = MY_HEAD

    # Debug board
    print (board)

    # Iterate over food and geese in order to compute distances for each one
    food_race = {}
    for food in observation.food:
        food_race[food] = {}
        for i in range(4):
            goose = observation.geese[i]
            if len(goose) == 0:
                continue
            food_race[food][i] = cell_distance(goose[0], food, configuration)

    # The best food is the least coveted
    best_food = None
    best_distance = float('inf')
    best_closest_geese = float('inf')
    for food in food_race:
        print ("-> Food on {}".format(row_col(food, configuration.columns)))
        my_distance = food_race[food][player_index]
        print (" - My distance is {}".format(my_distance))
        closest_geese = 0
        for goose_id in food_race[food]:
            if goose_id == player_index:
                continue
            if food_race[food][goose_id] <= my_distance:
                closest_geese += 1
        print (" - There are {} closest geese".format(closest_geese))
        if (closest_geese < best_closest_geese):
            best_food = food
            best_distance = my_distance
            best_closest_geese = closest_geese
            print ("  * This food is better")
        elif (closest_geese == best_closest_geese) and (my_distance <= best_distance):
            best_food = food
            best_distance = my_distance
            best_closest_geese = closest_geese
            print ("  * This food is better")

    # Now that the best food has been found, check if the movement towards it is safe.
    # Computes every available move and then check for move priorities.
    if len(player_goose) > 1:
        food_movements = move_towards(player_head, player_goose[1], best_food, configuration)
    else:
        food_movements = move_towards(player_head, player_head, best_food, configuration)
    all_movements = get_all_movements(player_head, configuration)
    # Excluding last movement reverse
    food_movements = [move for move in food_movements if move[2] != REVERSE_MOVE[last_move]]
    all_movements  = [move for move in all_movements if move[2] != REVERSE_MOVE[last_move]]
    print ("-> Available food moves: {}".format(food_movements))
    print ("-> All moves: {}".format(all_movements))

    # Trying to reach goal size of 4
    if (len(player_goose) < 4):

        # 1. Food movements that are safe and not closed
        for food_movement in food_movements:
            print ("Food movement {}".format(food_movement))
            if is_safe (food_movement, board) and not is_closed(food_movement, board):
                print ("It's safe! Let's move {}!".format(moves[food_movement[2]]))
                last_move = food_movement[2]
                return moves[food_movement[2]] # Move here

        # 2. Any movement safe and not closed
        for movement in all_movements:
            print ("Movement {}".format(movement))
            if is_safe (movement, board) and not is_closed(movement, board):
                print ("It's safe! Let's move {}!".format(moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 3. Food movements half safe and not closed
        for food_movement in food_movements:
            if is_half_safe (food_movement, board) and not is_closed(food_movement, board):
                print ("Food movement {} is half safe, I'm going {}!".format(food_movement, moves[food_movement[2]]))
                last_move = food_movement[2]
                return moves[food_movement[2]] # Move here

        # 4. Any movement half safe and not closed
        for movement in all_movements:
            if is_half_safe (movement, board) and not is_closed(movement, board):
                print ("Movement {} is half safe, I'm going {}!".format(movement, moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 5. Food movements that are safe
        for food_movement in food_movements:
            print ("Food movement {}".format(food_movement))
            if is_safe (food_movement, board):
                print ("It's safe! Let's move {}!".format(moves[food_movement[2]]))
                last_move = food_movement[2]
                return moves[food_movement[2]] # Move here

        # 6. Any movement safe
        for movement in all_movements:
            print ("Movement {}".format(movement))
            if is_safe (movement, board):
                print ("It's safe! Let's move {}!".format(moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 7. Food movements half safe
        for food_movement in food_movements:
            if is_half_safe (food_movement, board):
                print ("Food movement {} is half safe, I'm going {}!".format(food_movement, moves[food_movement[2]]))
                last_move = food_movement[2]
                return moves[food_movement[2]] # Move here

        # 8. Any movement half safe
        for movement in all_movements:
            if is_half_safe (movement, board):
                print ("Movement {} is half safe, I'm going {}!".format(movement, moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

    # Just trying to walk in circles
    else:

        # Delete food moves
        food_coordinates = []
        for food in food_race:
            x_food, y_food = row_col(food, configuration.columns)
            food_coordinates.append((x_food, y_food))
        available_moves = []
        for move in all_movements:
            for (x_food, y_food) in food_coordinates:
                if (move[0] != x_food) or (move[1] != y_food):
                    available_moves.append(move)

        # 1. Run in circles if you can
        circle_move = CIRCLE_MOVE[last_move]
        for move in available_moves:
            if (move[2] == circle_move) and (is_safe(move, board)) and not (is_closed(move, board)):
                last_move = move[2]
                return moves[move[2]]

        # 2. Any movement safe and not closed
        for movement in all_movements:
            print ("Movement {}".format(movement))
            if is_safe (movement, board) and not is_closed(movement, board):
                print ("It's safe! Let's move {}!".format(moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 3. Any movement half safe and not closed
        for movement in all_movements:
            if is_half_safe (movement, board) and not is_closed(movement, board):
                print ("Movement {} is half safe, I'm going {}!".format(movement, moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 4. Any movement safe
        for movement in all_movements:
            print ("Movement {}".format(movement))
            if is_safe (movement, board):
                print ("It's safe! Let's move {}!".format(moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

        # 5. Any movement half safe
        for movement in all_movements:
            if is_half_safe (movement, board):
                print ("Movement {} is half safe, I'm going {}!".format(movement, moves[movement[2]]))
                last_move = movement[2]
                return moves[movement[2]] # Move here

    # Finally, if all moves are unsafe, randomly pick one
    rand_pick = np.random.randint(4) + 1
    last_move = rand_pick
    print ("Yeah whatever, I'm going {}".format(moves[rand_pick]))
    return moves[rand_pick]