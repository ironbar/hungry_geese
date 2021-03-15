import os
import psutil
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import logging

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
x = np.random.uniform(size=(n_samples, input_features)).astype(np.float32)
y = np.random.uniform(size=(n_samples, output_features)).astype(np.float32)
logger.info('Input size: %.1f GB' % (x.nbytes/1e9))
logger.info('Data created')
log_ram_usage_and_available()

def generator(batch_size):
    while 1:
        start_idx = np.random.randint(0, x.shape[0] - batch_size, size=10000)
        for idx in start_idx:
            yield x[idx: idx+batch_size], y[idx: idx+batch_size]

gen = generator(8096)
logger.info('Created generator')
log_ram_usage_and_available()
model.fit(gen, epochs=3, steps_per_epoch=1236, callbacks=[LogRAM()])

"""
RAM usage is correct

2021-03-15 16:32:00,335 - __main__ - INFO - Model created
2021-03-15 16:32:00,335 - __main__ - INFO - Ram usage: 1.6 GB   available: 59.5 GB
2021-03-15 16:32:14,733 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 16:32:14,733 - __main__ - INFO - Data created
2021-03-15 16:32:14,733 - __main__ - INFO - Ram usage: 9.6 GB   available: 51.4 GB
2021-03-15 16:32:14,733 - __main__ - INFO - Created generator
2021-03-15 16:32:14,733 - __main__ - INFO - Ram usage: 9.6 GB   available: 51.4 GB
2021-03-15 16:32:14.760330: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 16:32:14.778596: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
Epoch 1/3
2021-03-15 16:32:14.939884: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:32:15.406806: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
1236/1236 [==============================] - 5s 4ms/step - loss: 0.2312
2021-03-15 16:32:20,008 - __main__ - INFO - Ram usage: 10.6 GB  available: 50.6 GB
Epoch 2/3
1236/1236 [==============================] - 5s 4ms/step - loss: 0.0844
2021-03-15 16:32:24,547 - __main__ - INFO - Ram usage: 10.6 GB  available: 50.6 GB
Epoch 3/3
1236/1236 [==============================] - 5s 4ms/step - loss: 0.0837
2021-03-15 16:32:29,084 - __main__ - INFO - Ram usage: 10.6 GB  available: 50.6 GB
"""