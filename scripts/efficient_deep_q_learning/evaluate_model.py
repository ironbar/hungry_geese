import os
import glob
import sys
import argparse
import logging
import time
import yaml
import subprocess
import tensorflow as tf

from hungry_geese.utils import configure_logging
from hungry_geese.utils import log_to_tensorboard

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
    template_path = conf['evaluate_template']
    n_matches = conf['n_matches_eval']
    evaluation_period = conf['evaluation_period']
    already_evaluated = set()
    tensorboard_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'logs_debug'))

    while 1:
        model_path = get_latest_model_path(model_dir)
        if model_path in already_evaluated:
            logger.info('Last model is already evaluated: %s' % model_path)
            time.sleep(1)
            continue
        epoch = get_epoch_from_model_path(model_path)
        if epoch % evaluation_period == 0 and epoch:
            logger.info('Going to evaluate: %s' % model_path)
            eval_results = evaluate_model(model_path, template_path, n_matches=n_matches)
            for key, value in eval_results.items():
                log_to_tensorboard(key, value, epoch, tensorboard_writer)
            already_evaluated.add(model_path)
        else:
            logger.info('Last model does not require evaluation: %s' % model_path)
            time.sleep(1)


def get_latest_model_path(model_dir):
    model_paths = sorted(glob.glob(os.path.join(model_dir, 'epoch*.h5')))
    return model_paths[-1]


def get_epoch_from_model_path(model_path):
    epoch = int(os.path.splitext(os.path.basename(model_path))[0].split('epoch_')[-1])
    return epoch


def evaluate_model(model_path, template_path, n_matches):
    logger.info('Evaluating model')
    command = 'python "%s" "%s" "%s" --n_matches %i' % (
        '/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/scripts/deep_q_learning/evaluate_model.py',
        model_path, template_path, n_matches)
    output = subprocess.getoutput(command).split('\n')
    elo_multi = int(output[-2].split('score: ')[-1])
    elo_single = int(output[-1].split('score: ')[-1])
    logger.info('Multi agent elo score: %i' % elo_multi)
    logger.info('Single agent elo score: %i' % elo_single)
    return dict(elo_multi=elo_multi, elo_single=elo_single, elo_mean=(elo_multi + elo_single)/2)


def parse_args(args):
    epilog = """
    """
    description = """
    The responsability of this script is to evaluate models and to save the results to tensorboar logs
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to yaml file with training configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()