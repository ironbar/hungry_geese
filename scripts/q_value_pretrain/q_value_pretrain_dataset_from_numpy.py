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
    logger.info('Creating tf.data.Dataset')
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_data[0], train_data[1], train_data[2]), train_data[3]))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_data[0], val_data[1], val_data[2]), val_data[3]))
    train_dataset = train_dataset.batch(conf['train_batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(conf['val_batch_size']).prefetch(tf.data.AUTOTUNE)
    log_ram_usage()

    # test sampling speed
    t0 = time.time()
    for _ in tqdm(train_dataset, desc='sampling speed test'):
        pass
    logger.info('It takes %.1f seconds to sample enough data for an epoch' % (time.time() - t0))

    training_model.fit(
        x=train_dataset, validation_data=val_dataset,
        callbacks=callbacks, **conf['fit_params'])


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