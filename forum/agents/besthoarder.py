"""
https://www.kaggle.com/leontkh/hard-coded-passive-aggressive-agent
"""

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from random import choice, sample, randint, seed

import time

def random_agent():
    seed(None,randint(1,2))
    step = choice([action for action in Action])
    seed(1)
    print(step.name,"bugged")
    return step.name


def translate(position: int, direction: Action, columns: int, rows: int):
    row = position // columns
    column = position % columns
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column


def adjacent_positions(position: int, columns: int, rows: int):
    return [
        translate(position, action, columns, rows)
        for action in Action
    ]


def min_distance(position: int, food: [int], columns: int):
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )

last_pos = {}

def agent(observation, configuration):
    global last_pos
    try:
        print("===ROUND ",observation.step+1,"===")
        time_start = time.perf_counter()
        observation = Observation(observation)
        configuration = Configuration(configuration)
        rows, columns = configuration.rows, configuration.columns

        food = observation.food
        geese = observation.geese
        player_index = observation.index
        my_goose = geese[player_index]
        my_tail = [my_goose[-1]]
        position = geese[observation.index][0]

        print("At row: ",position//11," column: ", position%11," food at: ", food)

        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose[0:-1]}
        # Opponent definition
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]
        # Don't move into tails of heads that are adjacent to food
        tails = {
            opponent[-1]
            for opponent in opponents
            for opponent_head in [opponent[0]]
            if any(
                adjacent_position in food
                # Head of opponent is adjacent to food so tail is not safe
                for adjacent_position in adjacent_positions(opponent_head, columns, rows)
            )
        }
        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }

        myself = [my_goose]

        keep_dist = {
            keep_dist
            for head_adjacent_position in head_adjacent_positions
            for keep_dist in adjacent_positions(head_adjacent_position, columns, rows)
        }

        """Hard coding movement here"""

        #MOVEMENT TIME
        if len(geese[player_index]) > 3 and len(geese[player_index]) < 9: #TURTLE POWER
            tactions = [
            action
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position in my_tail and
                new_position not in last_pos and
                new_position not in food #Checking for picking up food, which would kill my goose
            )
            ]
            if any(tactions): # CHeck if TURTLE POWER is valid
                new_tactions = [
                action
                for action in Action
                for new_position in [translate(position, action, columns, rows)]
                if (
                    new_position in my_tail and
                    new_position not in last_pos and
                    new_position not in food and
                    new_position not in head_adjacent_positions #Looking for goose walking into my tail
                )
                ]
                if any(new_tactions):
                    last_pos = {geese[player_index][0]}
                    print(new_tactions[0].name, "turtling")
                    seed(1)
                    return new_tactions[0].name

        # Checking if viable steps exist in next 4 turns
        possible_actions = {
            action: new_position5
            for action in Action
            for new_position1 in [translate(position, action, columns, rows)]
            if (
                new_position1 not in last_pos and
                new_position1 not in bodies and
                new_position1 not in tails
            )
            for action2 in Action
            for new_position2 in [translate(new_position1, action2, columns, rows)]
            if (
                new_position2 not in [geese[player_index][0]] and
                new_position2 not in bodies and
                new_position2 not in tails
            )
            for action3 in Action
            for new_position3 in [translate(new_position2, action3, columns, rows)]
            if (
                new_position3 not in [new_position1] and
                new_position3 not in bodies and
                new_position3 not in tails
            )
            for action4 in Action
            for new_position4 in [translate(new_position3, action4, columns, rows)]
            if (
                new_position4 not in [new_position2] and
                new_position4 not in bodies and
                new_position4 not in tails
            )
            for action5 in Action
            for new_position5 in [translate(new_position4, action5, columns, rows)]
            if (
                new_position5 not in [new_position3] and
                new_position5 not in bodies and
                new_position5 not in tails
            )
        }

        #Food turtling/Food hoarding/Passive Aggro Geese whatever you call it

        if len(geese[player_index]) > 7:
            factions = {
            action: min_distance(new_position, food, columns)
            for action in possible_actions
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in last_pos and
                new_position not in bodies and
                new_position not in tails and
                new_position not in food # Will avoid food and circle around it
            )
            if (
                new_position not in (food and my_tail)
            )
            }
            if any(factions):
                new_factions = {
                action: min_distance(new_position, food, columns)
                for action in possible_actions
                for new_position in [translate(position, action, columns, rows)]
                if (
                    new_position not in last_pos and
                    new_position not in bodies and
                    new_position not in tails and
                    new_position not in food and
                    new_position not in head_adjacent_positions
                )
                if (
                    new_position not in (food and my_tail)
                )
                }
                if any(new_factions):
                    last_pos = {geese[player_index][0]}
                    step = min(new_factions, key=new_factions.get)
                    print(step)
                    steps = [action for action in new_factions.keys() if(new_factions[action] == min(new_factions.values()))]
                    print(steps)
                    seed(None,randint(1,2))
                    step = choice(steps)
                    print(new_factions)

                    print(step)
                    print(step.name,"food hoarding")
                    seed(1)
                    return step.name

        #print(possible_actions)
        actions = {
            action: min_distance(new_position, food, columns)
            for action in possible_actions
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in last_pos and
                new_position not in bodies and
                new_position not in tails
            )
            if (
                new_position not in (food and my_tail)
            )
        }
        if any(actions):
            new_actions = {
                action: min_distance(new_position, food, columns)
                for action in possible_actions
                for new_position in [translate(position, action, columns, rows)]
                if (
                    new_position not in last_pos and
                    new_position not in bodies and
                    new_position not in tails and
                    new_position not in head_adjacent_positions
                )
                if (
                    new_position not in (food and my_tail)
                )
            }
            if any(new_actions):
                actions = new_actions

        #print(actions, '*')
        last_pos = {geese[player_index][0]}

        time_taken = time.perf_counter() - time_start
        #print(actions, time_taken)

        if any(actions):
            step = min(actions, key=actions.get)
            steps = [action for action in actions.keys() if(actions[action] == min(actions.values()))]
            seed(None,randint(1,2))
            step = choice(steps)
            print(step.name,"norm")
            seed(1)
            return step.name

    except Exception as e:
        print("ERROR:",e)
        return random_agent()

    return random_agent()