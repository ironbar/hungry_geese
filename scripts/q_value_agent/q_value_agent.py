import os
from hungry_geese.agents import QValueAgent

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

model_path = '/mnt/hdd0/Kaggle/hungry_geese/03_save_model/03_schedule_vs_random/model.h5'
model = tf.keras.models.load_model(model_path)
q_value_agent = QValueAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)
