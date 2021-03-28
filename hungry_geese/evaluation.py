import time
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from concurrent.futures import ProcessPoolExecutor
from kaggle_environments import evaluate


def play_matches_in_parallel(agents, sample_agents_func, max_workers=20, n_matches=1000,
                             running_on_notebook=True):
    """
    Plays n_matches in parallel using ProcessPoolExecutor

    Parameters
    -----------
    agents : dict
        Dictionary that matches name of the agents with the code
    sample_agents_func : func
        Function that returns random keys of the agents for playing a game
    """
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        matches_results = []
        matches_agents = []
        submits = []
        for i in range(n_matches):
            sampled_keys = sample_agents_func()
            submits.append(pool.submit(play_game, agents=[agents[key] for key in sampled_keys]))
            matches_agents.append(sampled_keys)
        monitor_progress(submits, running_on_notebook)

        matches_results = [submit.result()[0] for submit in submits]
    return matches_agents, matches_results

def play_game(agents):
    return evaluate("hungry_geese", agents=agents, num_episodes=1)

def monitor_progress(submits, running_on_notebook):
    if running_on_notebook:
        progress_bar = tqdm_notebook(total=len(submits))
    else:
        progress_bar = tqdm(total=len(submits))
    progress = 0
    while 1:
        time.sleep(1)
        current_progress = np.sum([submit.done() for submit in submits])
        if current_progress > progress:
            progress_bar.update(current_progress - progress)
            progress = current_progress
        if progress == len(submits):
            break
    time.sleep(0.1)
    progress_bar.close()