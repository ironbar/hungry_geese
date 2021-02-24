"""
https://www.kaggle.com/leontkh/hard-coded-passive-aggressive-agent
"""

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from random import choice, sample, seed, randint

def random_agent():
    return choice([action for action in Action]).name


def translate(position: int, direction: Action, columns: int, rows: int):
    row, column = row_col(position, columns)
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

last_pos = -1

def agent(observation, configuration):
    global last_pos
    observation = Observation(observation)
    configuration = Configuration(configuration)
    rows, columns = configuration.rows, configuration.columns

    food = observation.food
    geese = observation.geese

    player_index = observation.index
    my_goose = geese[player_index]
    my_tail = [my_goose[-1]]
    position = geese[observation.index][0]

    opponents = [
        goose
        for index, goose in enumerate(geese)
        if index != observation.index and len(goose) > 0
    ]

    # Don't move adjacent to any heads
    head_adjacent_positions = {
        opponent_head_adjacent
        for opponent in opponents
        for opponent_head in [opponent[0]]
        for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
    }
    # Don't move into any bodies
    bodies = {position for goose in geese for position in goose[0:-1]}
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

    # Move to the closest food
    position = geese[observation.index][0]
    if len(geese[player_index]) > 8: #HOARDING POWER
        factions = {
        action: min_distance(new_position, food, columns)
        for action in Action
        for new_position in [translate(position, action, columns, rows)]
        if (
            new_position not in last_pos and
            new_position not in bodies and
            new_position not in tails and
            new_position not in food
        )
        if (
            new_position not in (food and my_tail)
        )
        }
        if any(factions):
            new_factions = {
            action: min_distance(new_position, food, columns)
            for action in Action
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
                #print(step)
                steps = [action for action in new_factions.keys() if(new_factions[action] == min(new_factions.values()))]
                #print(steps)
                seed(None,randint(1,2))
                step = choice(steps)
                #print(new_factions)

                #print(step)
                #print(step.name,"food hoarding")
                seed(1)
                return step.name
    actions = {
        action: min_distance(new_position, food, columns)
        for action in Action
        for new_position in [translate(position, action, columns, rows)]
        if (
            new_position not in head_adjacent_positions and
            new_position not in bodies and
            new_position not in tails
        )
    }

    if any(actions):
        last_pos = {geese[player_index][0]}
        return min(actions, key=actions.get).name

    return random_agent()