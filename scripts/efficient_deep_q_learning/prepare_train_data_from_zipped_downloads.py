import os
import sys
import argparse
import logging
import pandas as pd
import json
import glob
import tempfile
from tqdm import tqdm

logger = logging.getLogger(__name__)

from hungry_geese.utils import configure_logging, get_timestamp

from play_matches_one_round import create_train_data


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    configure_logging(logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info('Created temporal folder: %s' % tmpdir)
        command = 'unzip -qq %s -d %s' % (args.zipped_matches, tmpdir)
        os.system(command)
        logger.info('Decompressed json files')


        os.makedirs(args.npz_output, exist_ok=True)
        json_filepaths = get_json_filepaths(tmpdir)


        matches = []
        zip_file_idx = int(os.path.splitext(os.path.basename(args.zipped_matches))[0])
        for json_filepath in tqdm(json_filepaths):
            with open(json_filepath, 'r') as f:
                match = json.load(f)
            matches.append(match)

            if len(matches) >= args.group_matches:
                output_path = os.path.join(args.npz_output, 'epoch_%05d_%s.npz' % (zip_file_idx, get_timestamp()))
                logger.info('Saving train data on: %s' % output_path)
                create_train_data(matches, args.reward_name, output_path)
                matches = []

def get_json_filepaths(folder):
    filepaths = sorted(glob.glob(os.path.join(folder, '*', '*.json')))
    # filepaths += sorted(glob.glob(os.path.join(folder, '*', '*', '*.json')))
    return filepaths


def parse_args(args):
    epilog = """
    python prepare_train_data_from_zipped_downloads.py \
    /mnt/hdd0/Kaggle/hungry_geese/downloads/zipped_matches/00.zip \
    v2_terminal_kill_and_grow_reward_2_-5_5_2_1 \
    /mnt/hdd0/Kaggle/hungry_geese/downloads/train_data 

    for i in {00..21}; do python prepare_train_data_from_zipped_downloads.py /mnt/hdd0/Kaggle/hungry_geese/downloads/zipped_matches/{i}.zip v2_terminal_kill_and_grow_reward_2_-5_5_2_1 /mnt/hdd0/Kaggle/hungry_geese/downloads/train_data & done
    """
    parser = argparse.ArgumentParser(
        description='Play matches in parallel using a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('zipped_matches', help='Path to the zip file with matches', type=str)
    parser.add_argument('reward_name', help='Name of the reward we want to use', type=str)
    parser.add_argument('npz_output', help='Path to the folder where npz files with train data will be saved', type=str)
    parser.add_argument('--group_matches', default=25, help='How many matches are grouped into a npz file', type=int)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()