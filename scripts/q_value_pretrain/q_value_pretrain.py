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
    model = simple_model()
    training_model = create_model_for_training(model)
    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = conf.get('learning_rate', 1e-3)
    training_model.compile('Adam', loss='mean_squared_error')
    log_ram_usage()

    callbacks = create_callbacks(conf, model_dir, conf['fit_params']['epochs'])
    log_ram_usage()

    training_model.fit(
        x=train_data[:3], y=train_data[-1], validation_data=(val_data[:3], val_data[-1]),
        callbacks=callbacks, **conf['fit_params'])

def load_data(filepath):
    logger.info('loading %s' % filepath)
    data = np.load(filepath)
    output = data['boards'], data['features'], data['actions'], data['rewards'].astype(np.float32)
    log_ram_usage()
    logger.info('data types: %s' % str([array.dtype for array in output]))
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
        # TensorBoardExtended(profile_batch=0, log_dir=os.path.join(model_folder, 'logs')),
        GarbageCollector(),
    ]
    # if 'security_checkpoint_period' in conf:
    #     callbacks.append(PeriodModelCheckpoint(
    #         model_folder,
    #         period=conf['security_checkpoint_period']))
    # if 'PeriodModelCheckpoint' in conf:
    #     logger.info('Adding PeriodModelCheckpoint with parameters: %s' % \
    #                  str(conf['PeriodModelCheckpoint']))
    #     callbacks.append(PeriodModelCheckpoint(
    #         folder=model_folder, **conf['PeriodModelCheckpoint']))
    # if 'ReduceLROnPlateau' in conf:
    #     callbacks.append(ReduceLROnPlateau(**conf['ReduceLROnPlateau']))
    # if 'EarlyStopping' in conf:
    #     logger.info('Adding EarlyStopping callback')
    #     callbacks.append(EarlyStopping(**conf['EarlyStopping']))
    # for key in conf:
    #     if 'ModelCheckpoint' in key and not 'PeriodModelCheckpoint' in key:
    #         logger.info('Adding %s callback' % key)
    #         conf[key]['filepath'] = os.path.join(model_folder, conf[key]['filename'])
    #         conf[key].pop('filename')
    #         callbacks.append(ModelCheckpoint(**conf[key]))
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