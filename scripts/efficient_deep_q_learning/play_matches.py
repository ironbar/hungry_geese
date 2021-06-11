import os
import glob
import sys
import argparse
import logging
import yaml
import subprocess
import tensorflow as tf
import time

from hungry_geese.utils import configure_logging, log_to_tensorboard, get_timestamp

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)


def main(args=None):
    configure_logging()
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)

    model_dir = os.path.dirname(os.path.realpath(args.config_path))
    tensorboard_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'logs'))

    while 1:
        model_path = get_latest_model_path(model_dir)
        epoch = get_epoch_from_model_path(model_path)
        train_data_path = os.path.join(model_dir, 'epoch_%05d_timestamp_%s.npz' % (epoch, get_timestamp()))
        play_results = play_matches(
            model_path=model_path,
            softmax_scale=conf['softmax_scale'],
            reward_name=conf['reward'],
            train_data_path=train_data_path,
            n_matches=conf['n_matches_play'],
            template_path=conf['play_template'],
            play_against_top_n=conf.get('play_against_top_n', 0),
            n_learning_agents=conf.get('n_learning_agents', 1),
            n_agents_for_experience=conf['n_agents_for_experience'])
        for key, value in play_results.items():
            log_to_tensorboard(key, value, epoch, tensorboard_writer)


def get_latest_model_path(model_dir):
    model_paths = sorted(glob.glob(os.path.join(model_dir, 'epoch*.h5')))
    return model_paths[-1]


def get_epoch_from_model_path(model_path):
    epoch = int(os.path.splitext(os.path.basename(model_path))[0].split('epoch_')[-1])
    return epoch


def play_matches(model_path, softmax_scale, reward_name, train_data_path, n_matches,
                 template_path, play_against_top_n, n_learning_agents, n_agents_for_experience):
    logger.info('Playing matches in parallel with: %s' % model_path)
    command = 'python "%s" "%s" %.1f %s "%s" "%s" --n_matches %i --play_against_top_n %i' % (
        '/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/scripts/efficient_deep_q_learning/faster_play_matches_one_round.py',
        model_path, softmax_scale, reward_name, train_data_path, template_path, n_matches, play_against_top_n)
    command += ' --n_learning_agents %i --n_agents_for_experience %i' % (n_learning_agents, n_agents_for_experience)
    t0 = time.time()
    output = eval(subprocess.getoutput(command).split('\n')[-1])
    output['play_time'] = time.time() - t0
    print(output)
    return output


def parse_args(args):
    epilog = """
    """
    description = """
    Play matches with the latests available model and saves metrics to tensorboard
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to yaml file with training configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()