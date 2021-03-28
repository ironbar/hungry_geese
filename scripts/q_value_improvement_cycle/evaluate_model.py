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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

from hungry_geese.evaluation import play_matches_in_parallel, monitor_progress
from hungry_geese.elo import EloRanking
from hungry_geese.definitions import INITIAL_ELO_RANKING, AGENT_TO_SCRIPT

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    simple_model_evaluation(args.model_path, n_matches=args.n_matches)


def simple_model_evaluation(model_path, n_matches=500):
    template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evaluation_template.py')
    with open(template_path, 'r') as f:
        text = f.read()
    text = text.replace('model_path', os.path.realpath(model_path))
    with tempfile.TemporaryDirectory() as tempdir:
        agent_filepath = os.path.join(tempdir, 'agent.py')
        with open(agent_filepath, 'w') as f:
            f.write(text)
        output = simple_agent_evaluation(agent_filepath, n_matches=n_matches)
    return output


def simple_agent_evaluation(agent_path, n_matches=500):
    """
    Computes single and multi agents scores and returns them

    Parameters
    ----------
    multi_agent_elo_score
    single_agent_elo_score
    """
    ret = evaluate_agent(
        {'q_value_pretrained': agent_path},
        INITIAL_ELO_RANKING.index.values.tolist()[:5],
        n_matches=n_matches, single_agent=False, max_workers=20)
    table_multi = compute_elo_ranking(*ret)
    ret = evaluate_agent(
        {'q_value_pretrained': agent_path},
        INITIAL_ELO_RANKING.index.values.tolist()[:5],
        n_matches=n_matches, single_agent=True, max_workers=20)
    table_single = compute_elo_ranking(*ret)
    print('Multi agent elo score: %i' % table_multi.loc['q_value_pretrained', 'ranking'])
    print('Single agent elo score: %i' % table_single.loc['q_value_pretrained', 'ranking'])
    return table_multi.loc['q_value_pretrained', 'ranking'], table_single.loc['q_value_pretrained', 'ranking']


def evaluate_agent(new_agent, adversary_agents, n_matches, single_agent=True, max_workers=20):
    agent_name = list(new_agent.keys())[0]
    reduced_agents_set = new_agent.copy()
    for adversary_agent in adversary_agents:
        reduced_agents_set[adversary_agent] = AGENT_TO_SCRIPT[adversary_agent]

    if single_agent:
        sample_agents_func = lambda: [agent_name] + np.random.choice(adversary_agents, 3, replace=False).tolist()
    else:
        def sample_agents_func():
            while 1:
                sampled_agents = [agent_name] + np.random.choice(adversary_agents + [agent_name], 3).tolist()
                if len(np.unique(sampled_agents)) >=2:
                    break
            return sampled_agents

    matches_agents, matches_results = play_matches_in_parallel(
        reduced_agents_set, sample_agents_func, n_matches=n_matches, max_workers=max_workers,
        running_on_notebook=False)
    return matches_agents, matches_results, reduced_agents_set


def compute_elo_ranking(matches_agents, matches_results, reduced_agents_set):
    initial_agent_elo = 1000
    for k in [32, 16, 8, 4, 2, 1]:
        agent_name = [name for name in reduced_agents_set if name not in INITIAL_ELO_RANKING][0]
        initial_ranking = INITIAL_ELO_RANKING.to_dict()['ranking']
        initial_ranking[agent_name] = initial_agent_elo
        initial_ranking = {key: initial_ranking[key] for key in reduced_agents_set}
        elo_ranking = EloRanking(initial_ranking, {agent_name}, k=k)
        for match_agents, match_results in zip(matches_agents, matches_results):
            elo_ranking.add_match(match_agents, match_results)
        initial_agent_elo = elo_ranking.summary().loc[agent_name, 'ranking']
    return elo_ranking.summary()


def parse_args(args):
    epilog = """
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('model_path', help='Path to the keras model')
    parser.add_argument('--n_matches', help='Number of matches that we want to run', type=int, default=500)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()