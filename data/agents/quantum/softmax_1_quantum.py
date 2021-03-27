import os
from hungry_geese.agents import SoftmaxAgent, EpsilonAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model_path = '/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/data/agents/quantum/best_model.h5'
model = tf.keras.models.load_model(model_path, compile=False)
model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
softmax_agent = SoftmaxAgent(model, scale=1)

def agent(obs, config):
    return softmax_agent(obs, config)
