import os
from hungry_geese.agents import QValueAgent, EpsilonAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model_path = '/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/data/agents/quantum/best_model.h5'
model = tf.keras.models.load_model(model_path, compile=False)
model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
q_value_agent = QValueAgent(model)
epsilon_agent = EpsilonAgent(q_value_agent, epsilon=0.05)

def agent(obs, config):
    return epsilon_agent(obs, config)
