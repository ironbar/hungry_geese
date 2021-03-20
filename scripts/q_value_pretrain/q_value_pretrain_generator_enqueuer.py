import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import logging
from functools import partial
import time
import random

from kaggle_environments import make
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from hungry_geese.model import simple_model, create_model_for_training
from hungry_geese.callbacks import (
    LogEpochTime, LogLearningRate, LogRAM, LogCPU, LogGPU, LogETA, GarbageCollector
)
from hungry_geese.utils import log_ram_usage, configure_logging
from hungry_geese.state import vertical_simmetry, horizontal_simmetry, player_simmetry

logger = logging.getLogger(__name__)



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_q_value(args)

def train_q_value(args):
    configure_logging()
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)
    model_dir = os.path.dirname(os.path.realpath(args.config_path))

    train_data = load_data(conf['train'])
    val_data = load_data(conf['val'])

    log_ram_usage()
    logger.info('creating model')
    model = simple_model(**conf['model_params'])
    model.summary()
    training_model = create_model_for_training(model)
    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = conf.get('learning_rate', 1e-3)
    training_model.compile(optimizer, loss='mean_squared_error')
    log_ram_usage()

    callbacks = create_callbacks(conf['callbacks'], model_dir, conf['fit_params']['epochs'])
    log_ram_usage()


    train_generator = generator(train_data, conf['train_batch_size'], data_augmentation=conf['data_augmentation'])
    val_generator = generator(val_data, conf['val_batch_size'])


    # Enqueuer
    generator_params = dict(use_multiprocessing=False)
    train_generator = tf.keras.utils.GeneratorEnqueuer(train_generator, **generator_params)
    val_generator = tf.keras.utils.GeneratorEnqueuer(val_generator, **generator_params)
    start_params = dict(workers=1, max_queue_size=10)
    train_generator.start(**start_params)
    val_generator.start(**start_params)

    # test sampling speed
    add_default_steps_if_missing(conf, train_data, val_data)
    sampling_speed_generator = train_generator.get()
    t0 = time.time()
    for _ in tqdm(range(conf['fit_params']['steps_per_epoch']), desc='sampling speed test'):
        next(sampling_speed_generator)
    logger.info('It takes %.1f seconds to sample enough data for an epoch' % (time.time() - t0))

    training_model.fit(
        x=train_generator.get(), validation_data=val_generator.get(),
        callbacks=callbacks, **conf['fit_params'])
    train_generator.stop()
    val_generator.stop()


def add_default_steps_if_missing(conf, train_data, val_data):
    conf['fit_params']['steps_per_epoch'] = conf['fit_params'].get(
        'steps_per_epoch', len(train_data[0])//conf['train_batch_size'])
    conf['fit_params']['validation_steps'] = conf['fit_params'].get(
        'validation_steps', len(val_data[0])//conf['val_batch_size'])
    logger.info('fit_params: %s' % str(conf['fit_params']))


def generator(train_data, batch_size, data_augmentation=False):
    idx_range = np.arange(len(train_data[0]))
    num_splits = len(idx_range)//batch_size
    logger.info('Looping over the dataset will take %i steps' % num_splits)
    while 1:
        np.random.shuffle(idx_range)
        for idx in range(num_splits):
            split_idx = idx_range[idx*batch_size:(idx+1)*batch_size]
            batch_data = [data[split_idx] for data in train_data]
            if data_augmentation:
                batch_data = apply_data_augmentation(batch_data)
            x = (batch_data[0], batch_data[1], batch_data[2])
            y = batch_data[3]
            yield (x, y)


def apply_data_augmentation(batch_data):
    if random.randint(0, 1):
        batch_data = vertical_simmetry(batch_data)
    if random.randint(0, 1):
        batch_data = horizontal_simmetry(batch_data)
    if random.uniform(0, 1) > 0.16:
        all_permutations = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        new_positions = all_permutations[random.randint(0, 4)]
        batch_data = player_simmetry(batch_data, new_positions)
    return batch_data


def load_data(filepath):
    logger.info('loading %s' % filepath)
    data = np.load(filepath)
    output = data['boards'], data['features'], data['actions'], data['rewards'].astype(np.float32)
    log_ram_usage()
    logger.info('data types: %s' % str([array.dtype for array in output]))
    logger.info('data shapes: %s' % str([array.shape for array in output]))
    return output


def create_callbacks(conf, model_folder, max_epochs):
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
        LogETA(max_epochs),
        tensorboard_callback,
        GarbageCollector(),
    ]
    if 'ReduceLROnPlateau' in conf:
        callbacks.append(ReduceLROnPlateau(**conf['ReduceLROnPlateau']))
    if 'EarlyStopping' in conf:
        logger.info('Adding EarlyStopping callback')
        callbacks.append(EarlyStopping(**conf['EarlyStopping']))
    for key in conf:
        if 'ModelCheckpoint' in key and not 'PeriodModelCheckpoint' in key:
            logger.info('Adding %s callback' % key)
            conf[key]['filepath'] = os.path.join(model_folder, conf[key]['filename'])
            conf[key].pop('filename')
            callbacks.append(ModelCheckpoint(**conf[key]))
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