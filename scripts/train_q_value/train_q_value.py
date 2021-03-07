import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from kaggle_environments import make
import tensorflow as tf

from hungry_geese.utils import log_to_tensorboard
from hungry_geese.agents import EpsilonAgent, QValueAgent
from hungry_geese.model import simple_model, create_model_for_training
from hungry_geese.state import combine_data, apply_all_simetries

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_q_value(args)

def train_q_value(args):
    with open(args.config_path, 'r') as f:
        conf = yaml.safe_load(f)

    model = simple_model()
    training_model = create_model_for_training(model)
    training_model.compile(optimizer='Adam', loss='mean_squared_error')
    base_agent = QValueAgent(model)
    epsilon_agent = EpsilonAgent(base_agent, epsilon=0.1)

    env = make('hungry_geese', configuration=dict(episodeSteps=200))
    trainer = env.train([None, "greedy", "greedy", "greedy"])
    configuration = env.configuration

    tensorboard_writer = tf.summary.create_file_writer(os.path.dirname(os.path.realpath(args.config_path)))
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
        log_to_tensorboard('epsilon_mean_reward', get_mean_reward(epsilon_data), epoch, tensorboard_writer)
        log_to_tensorboard('epsilon_mean_steps', get_mean_steps(epsilon_data), epoch, tensorboard_writer)

        all_data = agent_data + epsilon_data
        all_data = combine_data(all_data)
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