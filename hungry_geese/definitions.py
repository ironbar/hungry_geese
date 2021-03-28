"""
Commonly used paths, labels, and other stuff

Examples from Coldstart challenge

DATASET_PATH = '/media/guillermo/Data/DrivenData/Cold_Start'
TRAIN_PATH = os.path.join(DATASET_PATH, 'data', 'train.csv')
TEST_PATH = os.path.join(DATASET_PATH, 'data', 'test.csv')
METADATA_PATH = os.path.join(DATASET_PATH, 'data', 'meta.csv')
SUBMISSION_PATH = os.path.join(DATASET_PATH, 'data', 'submission_format.csv')
LIBRARY_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""
import os
import pandas as pd
import yaml

ACTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

ACTION_TO_IDX = {
    'NORTH': 0,
    'EAST': 1,
    'SOUTH': 2,
    'WEST': 3,
}

INITIAL_ELO_RANKING = pd.read_csv('/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/data/elo_ranking.csv', index_col='model')

with open('/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/data/agents.yml', 'r') as f:
    AGENT_TO_SCRIPT = yaml.safe_load(f)
