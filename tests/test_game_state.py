import pytest
from kaggle_environments import make

from hungry_geese.state import GameState
from hungry_geese.agents import SuicideAgent, ConstantAgentWithState, ConstantAgent

@pytest.mark.parametrize('steps_to_suicide', [2, 3, 4])
def test_game_state_two_geese(steps_to_suicide):
    env = make('hungry_geese', debug=True)
    env.reset(num_agents=2)
    agent = ConstantAgentWithState()
    ret = env.run([agent, SuicideAgent('NORTH', steps_to_suicide=steps_to_suicide)])
    # Give the last state as input
    agent(ret[-1][0]['observation'], env.configuration)
    assert len(ret) == steps_to_suicide + 2
    assert len(agent.state.rewards) == steps_to_suicide + 1
    assert agent.state.rewards[-1] == 1

    agent.reset()
    env.reset(num_agents=2)
    ret = env.run([agent, SuicideAgent('NORTH', steps_to_suicide=steps_to_suicide)])
    # Give the last state as input
    agent(ret[-1][0]['observation'], env.configuration)
    [print(step) for step in ret]
    assert len(ret) == steps_to_suicide + 2
    assert len(agent.state.rewards) == steps_to_suicide + 1
    assert agent.state.rewards[-1] == 1

def test_render_board():
    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy"])
    configuration = env.configuration
    obs = trainer.reset()
    state = GameState()
    board, features = state.update(obs, configuration)
    state.render_board(board)

def test_render_next_movements():
    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy"])
    configuration = env.configuration
    obs = trainer.reset()
    state = GameState()
    board, features = state.update(obs, configuration)
    state.render_next_movements(board)