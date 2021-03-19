import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import logging
from functools import partial
import time

from kaggle_environments import make
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from hungry_geese.model import simple_model, create_model_for_training
from hungry_geese.callbacks import (
    LogEpochTime, LogLearningRate, LogRAM, LogCPU, LogGPU, LogETA, GarbageCollector
)
from hungry_geese.utils import log_ram_usage, configure_logging

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


    train_generator = generator(train_data, conf['train_batch_size'])
    val_generator = generator(val_data, conf['val_batch_size'])


    # Dataset
    # (967816, 7, 11, 17), (967816, 9), (967816, 4), (967816,)]
    output_signature = (
        (tf.TensorSpec(shape=(None, 7, 11, 17), dtype=tf.float32), tf.TensorSpec(shape=(None, 9, ), dtype=tf.float32), tf.TensorSpec(shape=(None, 4, ), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, ), dtype=tf.float32)
    )
    train_generator = tf.data.Dataset.from_generator(partial(generator, train_data, conf['train_batch_size']), output_signature=output_signature)
    val_generator = tf.data.Dataset.from_generator(partial(generator, val_data, conf['val_batch_size']), output_signature=output_signature)
    train_generator = train_generator.prefetch(10)
    val_generator = val_generator.prefetch(10)

    add_default_steps_if_missing(conf, train_data, val_data)
    training_model.fit(
        x=train_generator, validation_data=val_generator,
        callbacks=callbacks, **conf['fit_params'])


def add_default_steps_if_missing(conf, train_data, val_data):
    conf['fit_params']['steps_per_epoch'] = conf['fit_params'].get(
        'steps_per_epoch', len(train_data[0])//conf['train_batch_size'])
    conf['fit_params']['validation_steps'] = conf['fit_params'].get(
        'validation_steps', len(val_data[0])//conf['val_batch_size'])
    logger.info('fit_params: %s' % str(conf['fit_params']))


def generator(train_data, batch_size):
    idx_range = np.arange(len(train_data[0]))
    num_splits = len(idx_range)//batch_size
    logger.info('Looping over the dataset will take %i steps' % num_splits)
    while 1:
        np.random.shuffle(idx_range)
        for idx in range(num_splits):
            split_idx = idx_range[idx*batch_size:(idx+1)*batch_size]
            x = (train_data[0][split_idx],
                 train_data[1][split_idx],
                 train_data[2][split_idx])
            y = train_data[3][split_idx]
            yield (x, y)


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