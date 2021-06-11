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
from hungry_geese.heuristic import get_certain_death_mask, adapt_mask_to_3d_action
from hungry_geese.reward import get_death_reward_from_name, get_reward


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    configure_logging()
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
            metrics = gather_metrics_from_matches(matches)
            create_train_data(matches, reward_name, output_path, agent_idx_range=list(range(n_agents_for_experience)))
            print(metrics) # for capturing this print with subprocess


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
        # logger.debug('Match steps: %i' % len(match))
        for agent_idx in agent_idx_range:
            train_data.append(create_match_data_for_training(match, agent_idx, state,
                                                             conf, reward_name))
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
    data = dict(
        boards=train_data[0],
        features=train_data[1],
        rewards=train_data[2],
        is_not_terminal=train_data[3],
        training_mask=train_data[4])
    update_data_propagating_death_reward(data)
    np.savez_compressed(output_path, **data)
    del state
    del train_data
    log_ram_usage()


def update_data_propagating_death_reward(data, discount_factor=1):
    indices = find_indices_of_all_terminal_and_learnable_actions(data)
    for step in indices:
        propagate_death_reward_backwards(step, data, discount_factor=discount_factor)


def find_indices_of_all_terminal_and_learnable_actions(data):
    indices = np.arange(len(data['rewards']))[((1 - np.max(data['is_not_terminal'], axis=1))*np.min(data['training_mask'], axis=1)) == 1]
    return indices


def propagate_death_reward_backwards(step, data, discount_factor=1):
    """
    Propagates death information from step to step -1, and
    continues to step -2 and so on if necessary
    """
    action_idx = np.arange(3)[(data['is_not_terminal'][step - 1]*data['training_mask'][step - 1]) == 1]
    if not action_idx.size: # I think this is very unlikely, but let's be cautious
        return
    data['is_not_terminal'][step - 1, action_idx] = 0
    data['rewards'][step - 1, action_idx] += np.max(data['rewards'][step])*discount_factor
    if np.max(data['is_not_terminal'][step - 1]) == 0 and np.min(data['training_mask'][step - 1]) == 1:
        propagate_death_reward_backwards(step - 1, data, discount_factor=discount_factor)


def create_match_data_for_training(match, agent_idx, state, conf, reward_name):
    """
    Creates data from the match for training

    Parameters
    ----------
    match : list
        The typical match returned by hungry geese game
    agent_idx : int
    state : GameState
    conf : dict
        Configuration of the game
    reward_name : str

    Returns
    -------
    boards, features, rewards, is_not_terminal, training_mask
    """
    first_action = match[1][agent_idx]['action']
    if first_action == 'SOUTH':
        state.reset(previous_action='SOUTH')
    else:
        state.reset()
    certain_death_masks, certain_death_rewards = [], []

    for step_idx, step in enumerate(match):
        if step_idx:
            action = step[agent_idx]['action']
            state.add_action(action)
        observation = step[0]['observation'].copy()
        observation['index'] = agent_idx
        state.update(observation, conf)

        is_end_of_match = not observation['geese'][agent_idx] or step_idx == len(match) - 1
        if is_end_of_match:
            break
        certain_death_masks.append(get_curated_certain_death_mask(observation, conf, state))
        certain_death_rewards.append(get_certain_death_reward(
            observation, step_idx, match, agent_idx, conf, reward_name))

    data = state.prepare_data_for_training()
    # create rewards for taken actions
    rewards = np.expand_dims(data[-1], axis=1)
    ohe_actions = data[-2]
    rewards = rewards*ohe_actions
    # add certain death reward
    certain_death_masks = np.array(certain_death_masks)
    certain_death_rewards = np.array(certain_death_rewards)
    rewards += np.clip(certain_death_masks - ohe_actions, 0, 1)*np.expand_dims(certain_death_rewards, axis=1)
    # is_not_terminal
    is_not_terminal = np.ones_like(rewards)
    is_not_terminal -= certain_death_masks
    is_not_terminal[-1] = 0
    # mask for training
    training_mask = np.zeros_like(rewards)
    training_mask += ohe_actions
    training_mask += certain_death_masks
    training_mask = np.clip(training_mask, 0, 1)

    return data[:2] + [rewards, is_not_terminal, training_mask]


def get_curated_certain_death_mask(observation, conf, state):
    certain_death_mask = get_certain_death_mask(observation, conf)
    certain_death_mask = adapt_mask_to_3d_action(certain_death_mask, state.get_last_action())
    certain_death_mask = np.floor(certain_death_mask)
    return certain_death_mask


def get_certain_death_reward(observation, step_idx, match, agent_idx, conf, reward_name):
    future_observation = _create_future_observation_where_agent_is_death(match, step_idx, agent_idx)
    return get_reward(future_observation, observation, conf, reward_name)


def _create_future_observation_where_agent_is_death(match, step_idx, agent_idx):
    future_observation = match[step_idx + 1][0]['observation'].copy()
    future_observation['index'] = agent_idx
    future_observation['geese'] = future_observation['geese'].copy()
    future_observation['geese'][agent_idx] = []
    return future_observation


def gather_metrics_from_matches(matches, agent_idx=0):
    """
    Computes metrics from the played matches

    - mean_match_steps
    - death ratio
    - mean goose size
    - position

    Parameters
    -----------
    matches : list of match
        List of matches with the format of hungry geese
    """
    match_steps, goose_size, final_score = [], [], []
    deaths = 0

    for match in tqdm(matches, desc='Gathering metrics from game data'):
        for step_idx, step in enumerate(match):
            goose = step[0]['observation']['geese'][agent_idx]
            if not goose:
                deaths += 1
                break
            goose_size.append(len(goose))
        match_steps.append(step_idx)
        rewards = [info['reward'] for info in step]
        final_score.append(get_final_score(rewards))

    final_score = np.array(final_score)
    metrics = {
        'mean_match_steps': np.mean(match_steps),
        'death_ratio': deaths/len(match_steps),
        'mean_goose_size': np.mean(goose_size),
        'mean_final_score': np.mean(final_score),
        'win_ratio': np.mean(final_score == 3),
    }
    return metrics


def get_final_score(rewards, agent_idx=0):
    """
    Returns a value between 0 and 3, being 0 the last position and 3 the agent winning the match
    """
    final_score = 0
    for idx, reward in enumerate(rewards):
        if idx == agent_idx:
            continue
        if reward < rewards[agent_idx]:
            final_score += 1
        elif reward == rewards[agent_idx]:
            final_score += 0.5
    return final_score


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