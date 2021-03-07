import pytest
from kaggle_environments import make

@pytest.fixture
def train_info():
    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy", "greedy", "greedy"])
    configuration = env.configuration
    obs = trainer.reset()
    return obs, configuration
