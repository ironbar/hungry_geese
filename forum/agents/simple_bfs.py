"""
https://www.kaggle.com/aatiffraz/simple-bfs-starter-agent
"""
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import numpy as np

directions = {0:'EAST', 1:'NORTH', 2:'WEST', 3:'SOUTH', 'EAST':0, 'NORTH':1, 'WEST':2, 'SOUTH':3}


def move(loc, direction):
    """Move the whole snake in the given direction"""
    global directions
    direction = directions[direction]
    new_loc = []
    if direction == 'EAST':
        new_loc.append(int(11*(loc[0]//11)  + (loc[0]%11 + 1)%11))
    elif direction == 'WEST':
        new_loc.append(int(11*(loc[0]//11) + (loc[0]%11 + 10)%11))
    elif direction == 'NORTH':
        new_loc.append(int(11*((loc[0]//11 + 6)%7) + loc[0]%11))
    else:
        new_loc.append(int(11*((loc[0]//11 + 1)%7) + loc[0]%11))
    if len(loc) == 1:
        return new_loc
    return new_loc + loc[:-1]


def greedy_choose(head, board):
    move_queue = []
    visited = [[[100, 'NA'] for _ in range(11)] for l in range(7)]
    visited[head//11][head%11][0] = 0

    for i in range(4):
        move_queue.append([head, [i]])

    while len(move_queue) > 0:
        now_move = move_queue.pop(0)

        next_step = move([now_move[0]], now_move[1][-1])[0]

        if board[next_step//11][next_step%11] < 0:
            continue

        if len(now_move[1]) < visited[next_step//11][next_step%11][0]:
            visited[next_step//11][next_step%11][0] = len(now_move[1])
            visited[next_step//11][next_step%11][1] = now_move[1][0]
            for i in range(4):
                move_queue.append([next_step, now_move[1] + [i]])

        if board[next_step//11][next_step%11] > 0:
            return now_move[1][0]
    return random.randint(0,3)



def agent(obs, conf):
    global directions

    obs = Observation(obs)
    conf = Configuration(conf)
    board = np.zeros((7, 11), dtype=int)

    #Obstacle-ize your opponents
    for ind, goose in enumerate(obs.geese):
        if ind == obs.index or len(goose) == 0:
            continue
        for direction in range(4):
            moved = move(goose, direction)
            for part in moved:
                board[part//11][part%11] -= 1

    #Obstacle-ize your body, except the last part
    if len(obs.geese[obs.index]) > 1:
        for k in obs.geese[obs.index][:-1]:
            board[k//11][k%11] -= 1

    #Count food only if there's no chance an opponent will meet you there
    for f in obs.food:
        board[f//11][f%11] += (board[f//11][f%11] == 0)

    return directions[greedy_choose(obs.geese[obs.index][0], board)]