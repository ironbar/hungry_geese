import pytest
from kaggle_environments import make

from hungry_geese.agents import SuicideAgent, ConstantAgentWithState, ConstantAgent

def test_game_state_two_geese(steps_to_suicide=2):
    env = make('hungry_geese', debug=True)
    env.reset(num_agents=2)
    agent = ConstantAgentWithState()
    ret = env.run([agent, SuicideAgent('NORTH', steps_to_suicide=steps_to_suicide)])
    assert len(ret) == steps_to_suicide + 2
