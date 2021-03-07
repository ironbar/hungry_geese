import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from kaggle_environments import make
import tensorflow as tf

from hungry_geese.utils import log_to_tensorboard, log_configuration_to_tensorboard
from hungry_geese.agents import EpsilonAgent, QValueAgent
from hungry_geese.model import simple_model, create_model_for_training
from hungry_geese.state import combine_data, apply_all_simetries
from hungry_geese.definitions import ACTIONS

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_q_value(args)

def train_q_value(args):
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)
    tensorboard_writer = tf.summary.create_file_writer(os.path.dirname(os.path.realpath(args.config_path)))
    log_configuration_to_tensorboard(conf, tensorboard_writer)

    model = simple_model()
    training_model = create_model_for_training(model)
    optimizer = tf.keras.optimizers.get(conf.get('optimizer', 'Adam'))
    optimizer.learning_rate = learning_rate=conf.get('learning_rate', 1e-3)
    training_model.compile(optimizer=optimizer, loss='mean_squared_error')
    base_agent = QValueAgent(model)
    epsilon_agent = EpsilonAgent(base_agent, epsilon=conf['epsilon'])

    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None] + conf['other_agents'])
    configuration = env.configuration

    initial_epoch = conf.get('initial_epoch', 0)
    for epoch in tqdm(range(initial_epoch, initial_epoch + conf['epochs']), desc='Epochs'):
        agent_data, epsilon_data = [], []
        for episode in range(conf['episodes_per_epoch']//2):
            play_episode(base_agent, trainer, configuration)
            agent_data.append(base_agent.state.prepare_data_for_training())
            play_episode(epsilon_agent, trainer, configuration)
            epsilon_data.append(base_agent.state.prepare_data_for_training())

        log_to_tensorboard('mean_reward', get_mean_reward(agent_data), epoch, tensorboard_writer)
        log_to_tensorboard('mean_steps', get_mean_steps(agent_data), epoch, tensorboard_writer)
        log_to_tensorboard('epsilon/mean_reward', get_mean_reward(epsilon_data), epoch, tensorboard_writer)
        log_to_tensorboard('epsilon/mean_steps', get_mean_steps(epsilon_data), epoch, tensorboard_writer)
        log_to_tensorboard('episodes', (epoch+1)*conf['episodes_per_epoch'], epoch, tensorboard_writer)

        agent_data = combine_data(agent_data)
        epsilon_data = combine_data(epsilon_data)
        log_action_distribution(agent_data[2], epoch, tensorboard_writer)
        log_action_distribution(epsilon_data[2], epoch, tensorboard_writer, prefix='epsilon_')

        all_data = combine_data([agent_data, epsilon_data])
        all_data = apply_all_simetries(all_data)
        ret = training_model.fit(x=all_data[:3], y=all_data[-1], epochs=1, verbose=False)
        log_to_tensorboard('loss', ret.history['loss'][-1], epoch, tensorboard_writer)

def play_episode(agent, trainer, configuration):
    agent.reset()
    obs = trainer.reset()
    done = False
    while not done:
        action = agent(obs, configuration)
        obs, reward, done, info = trainer.step(action)
    agent(obs, configuration)

def get_mean_reward(all_data):
    rewards = [data[3][0] for data in all_data]
    return np.mean(rewards)

def get_mean_steps(all_data):
    steps = [len(data[3]) for data in all_data]
    return np.mean(steps)

def log_action_distribution(actions, epoch, tensorboard_writer, prefix=''):
    action_distribution = np.sum(actions, axis=0)
    action_distribution /= np.sum(action_distribution)
    for idx, name in enumerate(ACTIONS):
        log_to_tensorboard('%saction_distribution/%s' % (prefix, name), action_distribution[idx], epoch, tensorboard_writer)


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