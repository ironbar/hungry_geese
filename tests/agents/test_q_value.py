import pytest
import random
import numpy as np
from kaggle_environments import make

from hungry_geese.agents import QValueAgent
from hungry_geese.utils import ACTIONS, opposite_action

random.seed(7)

class FakeModelUniform():
    def predict(self, *args, **kwargs):
        return np.ones((1, 4))

class FakeModelRandom():
    def predict(self, *args, **kwargs):
        return np.random.uniform(size=(1, 4))

@pytest.fixture
def train_info():
    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy", "greedy", "greedy"])
    configuration = env.configuration
    obs = trainer.reset()
    return obs, configuration

@pytest.mark.parametrize('model', [FakeModelRandom(), FakeModelUniform()])
def test_agent_makes_legal_actions(train_info, model):
    agent = QValueAgent(model)
    agent.reset()
    actions = [agent(*train_info) for _ in range(100)]
    for previous_action, action in zip(actions[:-1], actions[1:]):
        assert action != opposite_action(previous_action)
