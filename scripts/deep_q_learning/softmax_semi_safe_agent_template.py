import os
from hungry_geese.agents import SoftmaxSemiSafeAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model = tf.keras.models.load_model('model_path', compile=False)
softmax_agent = SoftmaxSemiSafeAgent(model, scale=softmax_scale)

def agent(obs, config):
    return softmax_agent(obs, config)