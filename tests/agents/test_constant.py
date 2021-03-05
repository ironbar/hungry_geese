import pytest

from hungry_geese.agents import ConstantAgent
from hungry_geese.utils import ACTIONS

@pytest.mark.parametrize('action', ACTIONS)
def test_constant_agent(action):
    agent = ConstantAgent(action)
    assert all(action == agent(None, None) for _ in range(5))
