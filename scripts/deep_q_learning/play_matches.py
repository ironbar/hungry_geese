import os
import sys
import argparse
import numpy as np
import tempfile
from tqdm import tqdm
import logging
from functools import partial
import time
import random
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from kaggle_environments import make

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

from hungry_geese.evaluation import play_matches_in_parallel, monitor_progress
from hungry_geese.elo import EloRanking
from hungry_geese.definitions import INITIAL_ELO_RANKING, AGENT_TO_SCRIPT
from hungry_geese.utils import log_ram_usage
from hungry_geese.state import GameState, combine_data
from hungry_geese.utils import configure_logging


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # configure_logging()
    simple_model_softmax_policy_data_generation(
        args.model_path, args.softmax_scale, args.output, args.n_matches, args.reward_name,
        template_path=args.template_path, play_against_top_n=args.play_against_top_n,
        n_learning_agents=args.n_learning_agents, n_agents_for_experience=args.n_agents_for_experience)


def simple_model_softmax_policy_data_generation(model_path, softmax_scale, output_path,
                                                n_matches, reward_name, template_path,
                                                play_against_top_n, n_learning_agents,
                                                n_agents_for_experience):
    # template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'play_template.py')
    with open(template_path, 'r') as f:
        text = f.read()
    text = text.replace('model_path', os.path.realpath(model_path))
    text = text.replace('softmax_scale', str(softmax_scale))
    with tempfile.TemporaryDirectory() as tempdir:
        agent_filepath = os.path.join(tempdir, 'agent.py')
        with open(agent_filepath, 'w') as f:
            f.write(text)
        if not play_against_top_n:
            matches = play_matches_in_parallel(agents=[agent_filepath]*4, n_matches=n_matches)
            create_train_data(matches, reward_name, output_path)
        else:
            script_path = os.path.realpath(__file__)
            repo_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
            current_elo_ranking = pd.read_csv(os.path.join(repo_path, 'data/elo_ranking.csv'), index_col='model')
            with open(os.path.join(repo_path, 'data/agents.yml'), 'r') as f:
                agents = yaml.safe_load(f)
            adversary_names = current_elo_ranking.index.values[:play_against_top_n].tolist()
            # logger.info('Playing against: %s' % str(adversary_names))
            # print('Playing against: %s' % str(adversary_names))
            adversaries = [agents[name] for name in adversary_names]
            sample_func = lambda: [agent_filepath]*n_learning_agents \
                + np.random.choice(adversaries, size=(4 - n_learning_agents)).tolist()
            matches = play_matches_in_parallel(agents=sample_func, n_matches=n_matches)
            create_train_data(matches, reward_name, output_path, agent_idx_range=list(range(n_agents_for_experience)))


def play_matches_in_parallel(agents, max_workers=20, n_matches=1000, running_on_notebook=False):
    """
    Plays n_matches in parallel using ProcessPoolExecutor

    Parameters
    -----------
    agents : list
        List of the agents that we will use for playing
    """
    log_ram_usage()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        matches_results = []
        submits = []
        for i in range(n_matches):
            if isinstance(agents, list):
                submits.append(pool.submit(play_game, agents=agents))
            elif callable(agents):
                submits.append(pool.submit(play_game, agents=agents()))
            else:
                raise TypeError(type(agents))
        monitor_progress(submits, running_on_notebook)
        matches_results = [submit.result() for submit in submits]
    log_ram_usage()
    return matches_results


def play_game(agents):
    env = make("hungry_geese")
    return env.run(agents=agents)


def create_train_data(matches_results, reward_name, output_path, agent_idx_range=None):
    """
    Creates train data without any simmetry

    Parameters
    ----------
    saved_games_paths : list of str
        Path to the games that we want to use
    reward_name : str
        Name of the reward function that we want to use
    output_path : str
        Path to the file were we are going to save the results
    max_workers : int
    agent_idx_range : list of int
        Idx of the agents we want to use for collecting data, if None all the agents
        will be used
    """
    env = make("hungry_geese")
    conf = env.configuration

    state = GameState(reward_name=reward_name, apply_reward_acumulation=False)
    train_data = []
    agent_idx_range = agent_idx_range or list(range(4))


    for _ in tqdm(range(len(matches_results)), desc='Creating game data'):
        match = matches_results[0]
        for agent_idx in agent_idx_range:
            state.reset()
            for step_idx, step in enumerate(match):
                observation = step[0]['observation'].copy()
                observation['index'] = agent_idx
                state.update(observation, conf)
                if step_idx:
                    state.add_action(step[agent_idx]['action'])
                if not observation['geese'][agent_idx]:
                    break
            data = state.prepare_data_for_training()
            is_not_terminal = np.ones_like(data[3])
            is_not_terminal[-1] = 0
            train_data.append(data + [is_not_terminal])
        del matches_results[0]

    log_ram_usage()
    logger.info('Going to combine the data')
    train_data = combine_data(train_data)
    log_ram_usage()
    logger.info('Size of the boards is %.1f GB (%s [%.1f GB])' % (
        train_data[0].nbytes/1e9,
        str([round(data.nbytes/1e9, 1) for data in train_data]),
        np.sum([data.nbytes/1e9 for data in train_data])))
    logger.info('Data shapes %s' % str([data.shape for data in train_data]))
    output_path = os.path.realpath(output_path)
    logger.info('Saving data on: %s' % output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, boards=train_data[0], features=train_data[1], actions=train_data[2], rewards=train_data[3], is_not_terminal=train_data[4])
    del state
    del train_data
    log_ram_usage()


def parse_args(args):
    epilog = """
    python scripts/deep_q_learning_v2/play_matches.py /mnt/hdd0/Kaggle/hungry_geese/models/31_iterating_over_softmax_policy/01_it1_2000_lr4e4/pretrained_model.h5 8 ranking_reward_-4_4 delete.npz "/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/scripts/deep_q_learning_v2/softmax_safe_agent_template.py" --n_matches 50
    """
    parser = argparse.ArgumentParser(
        description='Play matches in parallel using a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('model_path', help='Path to the keras model', type=str)
    parser.add_argument('softmax_scale', help='Scale of the softmax policy', type=float)
    parser.add_argument('reward_name', help='Name of the reward we want to use', type=str)
    parser.add_argument('output', help='Path to the npz file that will be created with the matches', type=str)
    parser.add_argument('template_path', help='Path to the python file with template for playing')
    parser.add_argument('--n_matches', help='Number of matches that we want to run', type=int, default=500)
    parser.add_argument('--play_against_top_n',
                        help='If given it will play against top n agents, otherwise it will do self-play',
                        type=int, default=0)
    parser.add_argument('--n_learning_agents',
                        help='How many learning agents to use, only when playing against top n agents',
                        type=int, default=1)
    parser.add_argument('--n_agents_for_experience',
                        help='How many agents to use for collecting experience, only when playing against top n agents',
                        type=int, default=1)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()