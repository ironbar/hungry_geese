import pytest
import random

from hungry_geese.agents import ConstantAgent, EpsilonAgent
from hungry_geese.utils import ACTIONS, opposite_action

random.seed(7)

@pytest.mark.parametrize('action', ['NORTH'])
@pytest.mark.parametrize('epsilon', [0, 0.5, 1])
def test_epsilon_agent_makes_legal_actions(action, epsilon):
    agent_base = ConstantAgent(action)
    agent = EpsilonAgent(agent_base, epsilon)
    agent.reset()
    actions = [agent(None, None) for _ in range(100)]
    for previous_action, action in zip(actions[:-1], actions[1:]):
        assert action != opposite_action(previous_action)

@pytest.mark.parametrize('action', ['NORTH', 'WEST'])
@pytest.mark.parametrize('epsilon, expected_action_probability', [
    (0, 1),
    (0.5, 0.5*(1+0.25)),
    (1, 0.25)
])
def test_epsilon_agent_random_action_probability(action, epsilon, expected_action_probability, n_runs=1000):
    agent_base = ConstantAgent(action)
    agent = EpsilonAgent(agent_base, epsilon)
    actions = [agent(None, None) for _ in range(n_runs)]
    matches = sum(action == _action for _action in actions)
    assert expected_action_probability == pytest.approx(matches/n_runs, abs=5e-2)
