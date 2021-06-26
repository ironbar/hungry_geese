import os
import sys
import argparse
import logging
import pandas as pd
import requests
import json
import glob

logger = logging.getLogger(__name__)

from hungry_geese.utils import configure_logging, get_timestamp

from play_matches_one_round import create_train_data


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    configure_logging(logging.INFO)
    logger.info('loading sorted episodes')
    sorted_episodes = pd.read_csv(args.sorted_episodes)
    os.makedirs(args.npz_output, exist_ok=True)
    os.makedirs(args.json_output, exist_ok=True)

    logger.info('starting to download matches')
    already_downloaded = get_already_downloaded_episodes(args.json_output)
    matches = []
    for episode_id in sorted_episodes.EpisodeId.values:
        if episode_id in already_downloaded:
            continue
        try:
            match = download_match(episode_id)
            already_downloaded.add(episode_id)
            matches.append(match)
            logger.info('Downloaded %i (%i steps)' % (episode_id, len(match)))
            with open(os.path.join(args.json_output, '%s.json' % episode_id), 'w') as outfile:
                json.dump(match, outfile)
        except IOError as exception:
            logger.error(str(exception))
        if len(matches) >= args.group_matches:
            output_path = os.path.join(args.npz_output, '%s.npz' % get_timestamp())
            logger.info('Saving train data on: %s' % output_path)
            create_train_data(matches, args.reward_name, output_path)


def download_match(episode_id):
    """ Downloads a match from kaggle website and returns the same output as running env.run() """
    base_url = "https://www.kaggle.com/requests/EpisodeService/"
    get_url = base_url + "GetEpisodeReplay"
    ret = requests.post(get_url, json = {"EpisodeId": int(episode_id)})
    if ret.status_code == 200:
        return json.loads(ret.json()['result']['replay'])['steps']
    else:
        raise IOError('Could not download %i status code: %i' % (episode_id, ret.status_code))


def get_already_downloaded_episodes(folder):
    already_downloaded = glob.glob(os.path.join(folder, '*.json'))
    already_downloaded = [int(os.path.splitext(os.path.basename(filepath))[0]) for filepath in already_downloaded]
    already_downloaded = set(already_downloaded)
    logger.info('Already downloaded %i matches' % len(already_downloaded))
    return already_downloaded


def parse_args(args):
    epilog = """
    python scripts/deep_q_learning_v2/play_matches.py /mnt/hdd0/Kaggle/hungry_geese/models/31_iterating_over_softmax_policy/01_it1_2000_lr4e4/pretrained_model.h5 8 ranking_reward_-4_4 delete.npz "/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/scripts/deep_q_learning_v2/softmax_safe_agent_template.py" --n_matches 50
    """
    parser = argparse.ArgumentParser(
        description='Play matches in parallel using a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('reward_name', help='Name of the reward we want to use', type=str)
    parser.add_argument('npz_output', help='Path to the folder where npz files with train data will be saved', type=str)
    parser.add_argument('json_output', help='Path to the folder where json files of the matches will be saved', type=str)
    parser.add_argument('--sorted_episodes', help='Path to the csv with the sorted episodes for download',
                        default='/mnt/hdd0/Kaggle/hungry_geese/downloads/sorted_episodes.csv')
    parser.add_argument('--group_matches', default=25, help='How many matches are grouped into a npz file', type=int)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()