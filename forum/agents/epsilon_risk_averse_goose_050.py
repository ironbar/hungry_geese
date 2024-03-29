"""
Copied from kaggle to see if it works better when parallelized
"""
import sys
import random
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation, Configuration, Action, adjacent_positions,
    min_distance,translate
)
from hungry_geese.agents import EpsilonAgent

sys.path.append('/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/forum/agents')
from risk_averse_goose import agent as base_agent

epsilon_agent = EpsilonAgent(base_agent, epsilon=0.50)

def agent(obs, config):
    return epsilon_agent(obs, config)
