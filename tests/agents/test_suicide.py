import pytest

from hungry_geese.agents import SuicideAgent
from hungry_geese.utils import ACTIONS, opposite_action

@pytest.mark.parametrize('action', ACTIONS)
@pytest.mark.parametrize('steps_to_suicide', [1, 2])
def test_suicide_agent(action, steps_to_suicide):
    agent = SuicideAgent(action, steps_to_suicide)
    assert all(action == agent(None, None) for _ in range(steps_to_suicide))
    assert opposite_action(action) == agent(None, None)
    agent.reset()
    assert action == agent(None, None)