import os
import logging
from hungry_geese.agents import QValueAgent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
model = tf.keras.models.load_model('model_path', compile=False)
q_value_agent = QValueAgent(model)
def agent(obs, config):
    return q_value_agent(obs, config)