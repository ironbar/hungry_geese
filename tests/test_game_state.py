import pytest
from kaggle_environments import make

from hungry_geese.agents import SuicideAgent, ConstantAgentWithState, ConstantAgent

@pytest.mark.parametrize('steps_to_suicide', [2, 3, 4])
def test_game_state_two_geese(steps_to_suicide):
    env = make('hungry_geese', debug=True)
    env.reset(num_agents=2)
    agent = ConstantAgentWithState()
    ret = env.run([agent, SuicideAgent('NORTH', steps_to_suicide=steps_to_suicide)])
    # Give the last state as input
    agent(ret[-1][0]['observation'], env.configuration)
    [print(step) for step in ret]
    assert len(ret) == steps_to_suicide + 2
    assert len(agent.state.rewards) == steps_to_suicide + 1
    assert agent.state.rewards[-1] == 1

    # agent.reset()
    # ret = env.run([agent, SuicideAgent('NORTH', steps_to_suicide=steps_to_suicide)])
    # # Give the last state as input
    # agent(ret[-1][0]['observation'], env.configuration)
    # assert len(ret) == steps_to_suicide + 2
    # assert len(agent.state.rewards) == steps_to_suicide + 1
    # assert agent.state.rewards[-1] == 1