import pytest
import random
import numpy as np

from hungry_geese.agents import QValueAgent
from hungry_geese.utils import ACTIONS, opposite_action

random.seed(7)

class FakeModelUniform():
    def predict(self, *args, **kwargs):
        return np.ones((1, 4))

class FakeModelRandom():
    def predict(self, *args, **kwargs):
        return np.random.uniform(size=(1, 4))

class FakeModelOhe():
    def predict(self, *args, **kwargs):
        ohe = np.zeros((1, 4))
        ohe[0, 0] = 1
        return ohe


@pytest.mark.parametrize('model', [FakeModelRandom(), FakeModelUniform(), FakeModelOhe()])
def test_agent_makes_legal_actions(train_info, model):
    agent = QValueAgent(model)
    agent.reset()
    actions = [agent(*train_info) for _ in range(100)]
    for previous_action, action in zip(actions[:-1], actions[1:]):
        assert action != opposite_action(previous_action)

@pytest.mark.parametrize('model', [FakeModelRandom(), FakeModelUniform()])
def test_agent_makes_random_actions(train_info, model):
    agent = QValueAgent(model)
    agent.reset()
    actions = [agent(*train_info) for _ in range(1000)]
    _, counts = np.unique(actions, return_counts=True)
    for count in counts:
        assert count > 200
        assert count < 300

@pytest.mark.parametrize('model', [FakeModelOhe()])
def test_agent_makes_constant_actions(train_info, model):
    agent = QValueAgent(model)
    agent.reset()
    actions = [agent(*train_info) for _ in range(100)]
    _, counts = np.unique(actions, return_counts=True)
    assert counts[0] == 100

@pytest.mark.parametrize('model', [FakeModelOhe()])
def test_agent_reset(train_info, model):
    agent = QValueAgent(model)
    agent.reset()
    assert not agent.state.history
    agent(*train_info)
    assert agent.state.history
    agent.reset()
    assert not agent.state.history

def test_agent_update_previous_action(train_info):
    agent = QValueAgent(FakeModelOhe())
    agent(*train_info)

    assert agent.previous_action == 'NORTH'
    assert agent.state.actions[-1] == 'NORTH'
    assert len(agent.state.actions) == 1

    agent.update_previous_action('SOUTH')
    assert agent.previous_action == 'SOUTH'
    assert agent.state.actions[-1] == 'SOUTH'
    assert len(agent.state.actions) == 1