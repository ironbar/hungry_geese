import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import logging

from kaggle_environments import make
import tensorflow as tf

from hungry_geese.model import simple_model, create_model_for_training

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_q_value(args)

def train_q_value(args):
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)
    model_dir = os.path.dirname(os.path.realpath(args.config_path))

    model = simple_model()
    training_model = create_model_for_training(model)
    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = conf.get('learning_rate', 1e-3)
    training_model.compile('Adam', loss='mean_squared_error')

    train_data = load_data(conf['train'])
    val_data = load_data(conf['val'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), profile_batch=0)
    ]

    training_model.fit(
        x=train_data[:3], y=train_data[-1], validation_data=(val_data[:3], val_data[-1]),
        callbacks=callbacks, **conf['fit_params'])

def load_data(filepath):
    logging.info('loading %s' % filepath)
    data = np.load(filepath)
    return data['boards'], data['features'], data['actions'], data['rewards']


def parse_args(args):
    epilog = """
    """
    parser = argparse.ArgumentParser(
        description='Train a model that learns q value function',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to yaml file with training configuration')
    return parser.parse_args(args)

if __name__ == '__main__':
    main()