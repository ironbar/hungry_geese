import os
import logging
from hungry_geese.agents import QValueSafeAgent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
model = tf.keras.models.load_model('model_path', compile=False)
model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
q_value_agent = QValueSafeAgent(model)
def agent(obs, config):
    return q_value_agent(obs, config)