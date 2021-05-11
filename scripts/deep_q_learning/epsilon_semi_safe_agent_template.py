import os
from hungry_geese.agents import EpsilonSemiSafeAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model = tf.keras.models.load_model('model_path', compile=False)
# this is a compromise, when using this agent scale is epsilon
softmax_agent = EpsilonSemiSafeAgent(model, epsilon=softmax_scale)

def agent(obs, config):
    return softmax_agent(obs, config)