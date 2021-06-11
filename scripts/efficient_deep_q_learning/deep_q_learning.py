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
from hungry_geese.utils import log_ram_usage, configure_logging, log_to_tensorboard
from hungry_geese.state import player_simmetry, get_ohe_opposite_actions, combine_data
from hungry_geese.loss import masked_mean_squared_error


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

    with tf.distribute.MirroredStrategy().scope():
        model, start_epoch = get_model(model_dir, conf)
        tensorboard_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'logs'))
        callbacks = create_callbacks(model_dir)
        log_ram_usage()
        data_enqueuer = create_data_enqueuer(conf)
        data_generator = data_enqueuer.get()

        sleep_between_epochs = conf.get('sleep_between_epochs', 0)
        for epoch_idx in range(start_epoch, conf['max_epochs']):
            logger.info('Starting epoch %i' % epoch_idx)
            train_model(model, conf, callbacks, epoch_idx, tensorboard_writer, data_generator)
            model_path = os.path.join(model_dir, 'epoch_%05d.h5' % epoch_idx)
            model.save(model_path, include_optimizer=False)
            if sleep_between_epochs:
                print('Sleeping %i seconds' % sleep_between_epochs)
                time.sleep(sleep_between_epochs)


def get_model(model_dir, conf):
    """
    Returns the model ready for training. If there is no model it creates a new one, otherwise it
    loads the last model

    Returns
    -------
    model, start_epoch
    """
    if get_last_saved_model(model_dir):
        model_path = get_last_saved_model(model_dir)
        logger.info('continuing training from: %s' % os.path.basename(model_path))
        start_epoch = int(model_path.split('epoch_')[-1].split('.h5')[0]) + 1
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        logger.info('creating model')
        model = globals()[conf['model']](**conf['model_params'])
        model_path = os.path.join(model_dir, 'epoch_00000.h5')
        model.save(model_path, include_optimizer=False)
        start_epoch = 1

    model.summary()

    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = conf.get('learning_rate', 1e-3)
    model.compile(optimizer, loss=masked_mean_squared_error)
    return model, start_epoch


def train_model(model, conf, callbacks, epoch_idx, tensorboard_writer, data_generator):
    model_input, model_output = get_model_input_and_output(model, conf, data_generator)

    log_ram_usage()
    initial_epoch = int(epoch_idx*conf['fit_epochs'])
    history = model.fit(
        x=model_input, y=model_output, batch_size=conf['train_batch_size'],
        callbacks=(callbacks), initial_epoch=initial_epoch,
        epochs=(initial_epoch + conf['fit_epochs']),
        **conf['fit_params']).history

    if 'random_matches' in conf:
        history['state_value'] = compute_state_value_evolution(
            model, os.path.join(conf['model_dir'], conf['random_matches']), conf['pred_batch_size'])
    history['n_train_files'] = len(_get_train_data_filepaths(conf['model_dir']))
    history['n_matches_played'] = history['n_train_files']*conf['n_matches_play']

    for key, value in history.items():
        log_to_tensorboard(key, np.mean(value), initial_epoch, tensorboard_writer)
    gc.collect()


def create_data_enqueuer(conf):
    logger.info('Creating data enqueuer')
    def simple_generator(conf):
        while 1:
            train_data = sample_train_data(conf['model_dir'], conf['aditional_files_for_training'],
                                           conf['epochs_to_sample_files_for_training'],
                                           conf['discount_factor'])
            yield train_data

    enqueuer = tf.keras.utils.GeneratorEnqueuer(simple_generator(conf))
    enqueuer.start(max_queue_size=1)
    return enqueuer


def get_model_input_and_output(model, conf, data_generator):
    """
    Samples train data and uses the model to compute the target, then applies data augmentation
    and returns the model input and output for training
    """
    train_data = next(data_generator)
    target = compute_q_learning_target(model, train_data, conf['discount_factor'], conf['pred_batch_size'])

    training_mask = train_data[2]
    model_output = np.concatenate([np.expand_dims(target, axis=2), np.expand_dims(training_mask, axis=2)], axis=2)
    train_data = train_data[:2] + [model_output, model_output]

    model_input = train_data[:2]
    model_output = train_data[2]
    return model_input, model_output


def sample_train_data(model_dir, aditional_files, epochs_to_sample, discount_factor):
    """
    New implementation does not give preference to new files, that needs to be implemented. Currently
    it simply samples from the last n files
    """
    # filepaths = sorted(glob.glob(os.path.join(model_dir, 'epoch*.npz')))
    # train_data = [load_data(filepaths[-1])]
    # if aditional_files:
    #     candidates = filepaths[-epochs_to_sample-1:-1]
    #     if candidates:
    #         if len(candidates) > aditional_files:
    #             samples = np.random.choice(candidates, aditional_files, replace=False)
    #         else:
    #             samples = candidates
    #         for sample in samples:
    #             logger.info('Loading aditional file for training: %s' % sample)
    #             train_data += [load_data(sample, verbose=False)]
    aditional_files = aditional_files + 1
    filepaths = _get_train_data_filepaths(model_dir)
    train_data = []

    candidates = filepaths[-epochs_to_sample-1:]
    samples = np.random.choice(candidates, aditional_files, replace=len(candidates) < aditional_files)
    for sample in samples:
        data = load_data(sample, verbose=False, discount_factor=discount_factor)
        data = random_data_augmentation(data)
        train_data.append(data)
    return combine_data(train_data)


def _get_train_data_filepaths(model_dir):
    filepaths = sorted(glob.glob(os.path.join(model_dir, 'epoch*.npz')))
    return filepaths


def random_data_augmentation(data):
    data = random_horizontal_simmetry(data)
    data = player_simmetry(data, np.random.choice(range(3), 3, replace=False))
    return data


def random_horizontal_simmetry(data):
    """ Randomly applies horizontal simmetry on board, training_mask, rewards and is_not_terminal"""
    if random.randint(0, 1):
        return data
    data = data[0][:, :, ::-1], data[1], data[2][:, ::-1], data[3][:, ::-1], data[4][:, ::-1]
    return data


def load_data(filepath, verbose=True, discount_factor=1):
    """
    Returns
    --------
    board, features, training_mask, rewards, is_not_terminal
    [(None, 11, 11, 17), (None, 9), (None, 3), (None, 3), (None, 3)]
    """
    if verbose: logger.info('loading %s' % filepath)
    data = np.load(filepath)
    output = [data['boards'], data['features'], data['training_mask'], data['rewards'], data['is_not_terminal']]
    if verbose: log_ram_usage()
    if verbose: logger.info('data types: %s' % str([array.dtype for array in output]))
    if verbose: logger.info('data shapes: %s' % str([array.shape for array in output]))
    return output


def compute_q_learning_target(model, train_data, discount_factor, batch_size):
    reward = train_data[3]
    is_not_terminal = train_data[4]
    state_value = np.expand_dims(compute_state_value(model, train_data, batch_size), axis=1)
    target = reward + is_not_terminal*discount_factor*state_value
    return target


def compute_state_value(model, train_data, batch_size):
    pred_q_values = model.predict(train_data[:2], batch_size=batch_size, verbose=1)
    pred_q_values[:-1] = pred_q_values[1:] # we use the next state for the prediction
    state_value = np.max(pred_q_values, axis=1)
    return state_value


def compute_state_value_evolution(model, data_path, batch_size):
    data = load_data(data_path)
    state_value = compute_state_value(model, data, batch_size)
    return np.mean(state_value)


def get_last_saved_model(model_dir):
    # TODO: move to library
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
    callbacks = [
        LogEpochTime(),
        LogLearningRate(),
        LogRAM(),
        LogCPU(),
        LogGPU(),
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