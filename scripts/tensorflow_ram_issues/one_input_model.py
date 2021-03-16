import os
import psutil
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import logging
import gc
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# tf.compat.v1.disable_eager_execution()


def log_ram_usage_and_available():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss/1e9
    stats = psutil.virtual_memory()  # returns a named tuple
    ram_available = getattr(stats, 'available')/1e9
    logger.info('Ram usage: %.1f GB\tavailable: %.1f GB' % (ram_usage, ram_available))

class LogRAM(keras.callbacks.Callback):
    @staticmethod
    def on_epoch_end(epoch, logs=None):
        log_ram_usage_and_available()

input_features, output_features = 200, 1
inputs = keras.layers.Input((input_features,))
outputs = keras.layers.Dense(output_features)(inputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile('Adam', loss='mean_squared_error')
#model.summary()
logger.info('Model created')
log_ram_usage_and_available()

n_samples = int(1e7)
x = np.ones((n_samples, input_features), dtype=np.float32)
y = np.ones((n_samples, output_features), dtype=np.float32)
logger.info('Input size: %.1f GB' % (x.nbytes/1e9))
logger.info('Data created')
for _ in range(3):
    time.sleep(1)
    log_ram_usage_and_available()
model.fit(x, y, epochs=3, batch_size=8096, callbacks=[LogRAM()], shuffle=False)
log_ram_usage_and_available()
for _ in range(3):
    gc.collect()
    time.sleep(1)
    log_ram_usage_and_available()

# def create_dataset(n_samples):
#     x = np.ones((n_samples, input_features), dtype=np.float32)
#     y = np.ones((n_samples, output_features), dtype=np.float32)
#     logger.info('Input size: %.1f GB' % (x.nbytes/1e9))
#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(8096).prefetch(10)
#     del x
#     del y
#     return dataset

# dataset = create_dataset(int(1e7))
# logger.info('Data created')
# log_ram_usage_and_available()
# gc.collect()
# log_ram_usage_and_available()
# model.fit(dataset, epochs=3, callbacks=[LogRAM()])
# log_ram_usage_and_available()
# for _ in range(3):
#     gc.collect()
#     log_ram_usage_and_available()