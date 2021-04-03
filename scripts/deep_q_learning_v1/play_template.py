import os
from hungry_geese.agents import SoftmaxAgent, EpsilonAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model = tf.keras.models.load_model('model_path', compile=False)
model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
softmax_agent = SoftmaxAgent(model, scale=softmax_scale)

def agent(obs, config):
    return softmax_agent(obs, config)