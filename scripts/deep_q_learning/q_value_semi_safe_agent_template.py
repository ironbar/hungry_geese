import os
import logging
from hungry_geese.agents import QValueSemiSafeAgent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
model = tf.keras.models.load_model('model_path', compile=False)
q_value_agent = QValueSemiSafeAgent(model)
def agent(obs, config):
    return q_value_agent(obs, config)