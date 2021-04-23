import os
import glob
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import logging
from functools import partial
import time
import random
import subprocess
import gc

from kaggle_environments import make
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from hungry_geese.model import simple_model, torus_model, create_model_for_training
from hungry_geese.callbacks import (
    LogEpochTime, LogLearningRate, LogRAM, LogCPU, LogGPU, LogETA, GarbageCollector, LogConstantValue
)
from hungry_geese.utils import log_ram_usage, configure_logging
from hungry_geese.state import apply_all_simetries, get_ohe_opposite_actions, combine_data

logger = logging.getLogger(__name__)



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    deep_q_learning(args)


def deep_q_learning(args):
    configure_logging()
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)
    model_dir = os.path.dirname(os.path.realpath(args.config_path))
    conf['model_dir'] = model_dir

    if 'pretrained_model' in conf or get_last_saved_model(model_dir):
        if 'pretrained_model' in conf:
            start_epoch = 0
            model_path = os.path.join(model_dir, conf['pretrained_model'])
            logger.info('loading pretrained model: %s' % conf['pretrained_model'])
        else:
            model_path = get_last_saved_model(model_dir)
            logger.info('continuing training from: %s' % os.path.basename(model_path))
            start_epoch = int(model_path.split('epoch_')[-1].split('.h5')[0]) + 1
        training_model = tf.keras.models.load_model(model_path)
        model = tf.keras.models.Model(inputs=training_model.inputs[:2], outputs=training_model.get_layer('action').output)
    else:
        logger.info('creating model')
        model = globals()[conf['model']](**conf['model_params'])
        model.summary()
        training_model = create_model_for_training(model)
        model_path = os.path.join(model_dir, 'random.h5')
        training_model.save(model_path, include_optimizer=False)
        start_epoch = 0

    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = conf.get('learning_rate', 1e-3)
    training_model.compile(optimizer, loss='mean_squared_error')
    callbacks = create_callbacks(model_dir)
    log_ram_usage()

    other_metrics = dict()
    for epoch_idx in range(start_epoch, conf['max_epochs']):
        other_metrics['n_matches'] = (epoch_idx+1)*conf['n_matches_play']
        logger.info('Starting epoch %i' % epoch_idx)
        train_data_path = os.path.join(model_dir, 'epoch_%04d.npz' % epoch_idx)
        play_matches(model_path, conf['softmax_scale'], conf['reward'], train_data_path, conf['n_matches_play'],
                     template_path=conf['play_template'], play_against_top_n=conf.get('play_against_top_n', 0),
                     n_learning_agents=conf.get('n_learning_agents', 1), n_agents_for_experience=conf['n_agents_for_experience'])
        train_model(training_model, model, conf, callbacks, epoch_idx, other_metrics, n_agents_for_experience=conf['n_agents_for_experience'])
        model_path = os.path.join(model_dir, 'epoch_%04d.h5' % epoch_idx)
        training_model.save(model_path, include_optimizer=False)
        if epoch_idx % conf.get('evaluation_period', 1) == 0 and epoch_idx:
            other_metrics = evaluate_model(model_path, conf['evaluate_template'], conf['n_matches_eval'])
        else:
            other_metrics = dict()


def train_model(training_model, model, conf, callbacks, epoch_idx, other_metrics, n_agents_for_experience):
    if 'random_matches' in conf:
        other_metrics['state_value'] = compute_state_value_evolution(
            model, os.path.join(conf['model_dir'], conf['random_matches']), conf['pred_batch_size'])

    train_data, steps_last_file = sample_train_data(
        conf['model_dir'], conf['aditional_files_for_training'], conf['epochs_to_sample_files_for_training'])
    other_metrics['mean_match_steps'] = steps_last_file/n_agents_for_experience/conf['n_matches_play']
    target = compute_q_learning_target(model, train_data, conf['discount_factor'], conf['pred_batch_size'])
    train_data = train_data[:3] + [target]
    train_data = apply_all_simetries(train_data)

    log_ram_usage()
    initial_epoch = int(epoch_idx*conf['fit_epochs'])
    aditional_callbacks = [LogConstantValue(key, value) for key, value in other_metrics.items()]
    training_model.fit(
        x=train_data[:3], y=train_data[3], batch_size=conf['train_batch_size'],
        callbacks=(aditional_callbacks + callbacks), initial_epoch=initial_epoch,
        epochs=(initial_epoch + conf['fit_epochs']),
        **conf['fit_params'])

    del aditional_callbacks
    del train_data
    gc.collect()


def play_matches(model_path, softmax_scale, reward_name, train_data_path, n_matches,
                 template_path, play_against_top_n, n_learning_agents, n_agents_for_experience):
    logger.info('Playing matches in parallel')
    command = 'python play_matches.py "%s" %.1f %s "%s" "%s" --n_matches %i --play_against_top_n %i' % (
        model_path, softmax_scale, reward_name, train_data_path, template_path, n_matches, play_against_top_n)
    command += ' --n_learning_agents %i --n_agents_for_experience %i' % (n_learning_agents, n_agents_for_experience)
    os.system(command)


def evaluate_model(model_path, template_path, n_matches):
    logger.info('Evaluating model')
    command = 'python "%s" "%s" "%s" --n_matches %i' % (
        'evaluate_model.py',
        model_path, template_path, n_matches)
    output = subprocess.getoutput(command).split('\n')
    elo_multi = int(output[-2].split('score: ')[-1])
    elo_single = int(output[-1].split('score: ')[-1])
    logger.info('Multi agent elo score: %i' % elo_multi)
    logger.info('Single agent elo score: %i' % elo_single)
    return dict(elo_multi=elo_multi, elo_single=elo_single)


def sample_train_data(model_dir, aditional_files, epochs_to_sample):
    filepaths = sorted(glob.glob(os.path.join(model_dir, 'epoch*.npz')))
    train_data = [load_data(filepaths[-1])]
    steps_last_file = len(train_data[0][0])
    if aditional_files:
        candidates = filepaths[-epochs_to_sample-1:-1]
        if candidates:
            if len(candidates) > aditional_files:
                samples = np.random.choice(candidates, aditional_files, replace=False)
            else:
                samples = candidates
            for sample in samples:
                logger.info('Loading aditional file for training: %s' % sample)
                train_data += [load_data(sample, verbose=False)]
    return combine_data(train_data), steps_last_file


def load_data(filepath, verbose=True):
    if verbose: logger.info('loading %s' % filepath)
    data = np.load(filepath)
    output = [data['boards'], data['features'], data['actions'], data['rewards'], data['is_not_terminal']]
    log_ram_usage()
    if verbose: logger.info('data types: %s' % str([array.dtype for array in output]))
    if verbose: logger.info('data shapes: %s' % str([array.shape for array in output]))
    return output


def compute_q_learning_target(model, train_data, discount_factor, batch_size):
    reward = train_data[3]
    is_not_terminal = train_data[4]
    state_value = compute_state_value(model, train_data, batch_size)
    target = reward + is_not_terminal*discount_factor*state_value
    return target


def compute_state_value(model, train_data, batch_size):
    actions = train_data[2]
    opposite_actions = get_ohe_opposite_actions(actions)
    pred_q_values = model.predict(train_data[:2], batch_size=batch_size, verbose=1)
    pred_q_values[:-1] = pred_q_values[1:] # we use the next state for the prediction
    state_value = np.max(pred_q_values - opposite_actions*1e3, axis=1)
    return state_value


def compute_state_value_evolution(model, data_path, batch_size):
    data = load_data(data_path)
    state_value = compute_state_value(model, data, batch_size)
    return np.mean(state_value)


def get_last_saved_model(model_dir):
    models = sorted(glob.glob(os.path.join(model_dir, 'epoch*.h5')))
    if models:
        return models[-1]
    return False


def create_callbacks(model_folder):
    """
    Params
    -------
    conf : dict
        Configuration of the callbacks from the train configuration file
    """
    logger.info('creating callbacks')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(model_folder, 'logs'), profile_batch=0)
    tensorboard_callback._supports_tf_logs = False
    callbacks = [
        LogEpochTime(),
        LogLearningRate(),
        LogRAM(),
        LogCPU(),
        LogGPU(),
        tensorboard_callback,
        GarbageCollector(),
    ]
    return callbacks


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