import numpy as np
import tensorflow.keras as keras
import time

def measure_model_speed(model, batch_size, n_predictions):
    x = [np.zeros((batch_size, 7, 11, 17), dtype=np.float32), np.zeros((batch_size, 9), dtype=np.float32)]
    model.predict_on_batch(x)
    t0 = time.time()
    for _ in range(n_predictions):
        model.predict_on_batch(x)
    mean_prediction_time = 1e6*(time.time() - t0)/n_predictions/batch_size
    print('Mean prediction time for batch size %i is: %.2f microseconds (total time %.2f seconds)' % (batch_size, mean_prediction_time, time.time() - t0))
    return mean_prediction_time

model = keras.models.load_model('/mnt/hdd0/Kaggle/hungry_geese/models/32_automated_iteration/11_continue_with_scale_8/epoch_0347.h5', compile=False)
#model = keras.models.load_model('/mnt/hdd0/Kaggle/hungry_geese/models/32_automated_iteration/06_200play_200eval_lr2e4_scale4_from_zero/epoch_0000.h5', compile=False)
model = keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)

batch_size_range = (2**np.arange(0, 14)).astype(np.int)
mean_prediction_times = []
for batch_size in batch_size_range:
    mean_prediction_times.append(measure_model_speed(model, batch_size=batch_size, n_predictions=100))

"""
/mnt/hdd0/Kaggle/hungry_geese/models/32_automated_iteration/11_continue_with_scale_8/epoch_0347.h5
cpu
Mean prediction time for batch size 1 is: 3537.48 microseconds (total time 0.35 seconds)
Mean prediction time for batch size 2 is: 2520.23 microseconds (total time 0.50 seconds)
Mean prediction time for batch size 4 is: 1232.19 microseconds (total time 0.49 seconds)
Mean prediction time for batch size 8 is: 474.26 microseconds (total time 0.38 seconds)
Mean prediction time for batch size 16 is: 246.64 microseconds (total time 0.39 seconds)
Mean prediction time for batch size 32 is: 160.54 microseconds (total time 0.51 seconds)
Mean prediction time for batch size 64 is: 102.32 microseconds (total time 0.65 seconds)
Mean prediction time for batch size 128 is: 56.96 microseconds (total time 0.73 seconds)
Mean prediction time for batch size 256 is: 35.52 microseconds (total time 0.91 seconds)
Mean prediction time for batch size 512 is: 21.68 microseconds (total time 1.11 seconds)
Mean prediction time for batch size 1024 is: 14.25 microseconds (total time 1.46 seconds)
Mean prediction time for batch size 2048 is: 10.75 microseconds (total time 2.20 seconds)
Mean prediction time for batch size 4096 is: 7.71 microseconds (total time 3.16 seconds)
Mean prediction time for batch size 8192 is: 10.38 microseconds (total time 8.50 seconds)

gpu
Mean prediction time for batch size 1 is: 6849.81 microseconds (total time 0.69 seconds)
Mean prediction time for batch size 2 is: 3949.37 microseconds (total time 0.79 seconds)
Mean prediction time for batch size 4 is: 1653.99 microseconds (total time 0.66 seconds)
Mean prediction time for batch size 8 is: 880.14 microseconds (total time 0.70 seconds)
Mean prediction time for batch size 16 is: 339.69 microseconds (total time 0.54 seconds)
Mean prediction time for batch size 32 is: 149.21 microseconds (total time 0.48 seconds)
Mean prediction time for batch size 64 is: 115.64 microseconds (total time 0.74 seconds)
Mean prediction time for batch size 128 is: 55.70 microseconds (total time 0.71 seconds)
Mean prediction time for batch size 256 is: 25.24 microseconds (total time 0.65 seconds)
Mean prediction time for batch size 512 is: 19.70 microseconds (total time 1.01 seconds)
Mean prediction time for batch size 1024 is: 12.09 microseconds (total time 1.24 seconds)
Mean prediction time for batch size 2048 is: 7.76 microseconds (total time 1.59 seconds)
Mean prediction time for batch size 4096 is: 6.47 microseconds (total time 2.65 seconds)
Mean prediction time for batch size 8192 is: 7.42 microseconds (total time 6.07 seconds)
"""


"""
/mnt/hdd0/Kaggle/hungry_geese/models/32_automated_iteration/06_200play_200eval_lr2e4_scale4_from_zero/epoch_0000.h5
cpu
Mean prediction time for batch size 1 is: 3809.49 microseconds (total time 0.38 seconds)
Mean prediction time for batch size 2 is: 2728.85 microseconds (total time 0.55 seconds)
Mean prediction time for batch size 4 is: 1030.84 microseconds (total time 0.41 seconds)
Mean prediction time for batch size 8 is: 688.59 microseconds (total time 0.55 seconds)
Mean prediction time for batch size 16 is: 372.12 microseconds (total time 0.60 seconds)
Mean prediction time for batch size 32 is: 187.96 microseconds (total time 0.60 seconds)
Mean prediction time for batch size 64 is: 103.59 microseconds (total time 0.66 seconds)
Mean prediction time for batch size 128 is: 62.79 microseconds (total time 0.80 seconds)
Mean prediction time for batch size 256 is: 52.73 microseconds (total time 1.35 seconds)
Mean prediction time for batch size 512 is: 30.17 microseconds (total time 1.54 seconds)
Mean prediction time for batch size 1024 is: 22.19 microseconds (total time 2.27 seconds)
Mean prediction time for batch size 2048 is: 13.84 microseconds (total time 2.83 seconds)
Mean prediction time for batch size 4096 is: 13.82 microseconds (total time 5.66 seconds)
Mean prediction time for batch size 8192 is: 15.70 microseconds (total time 12.86 seconds)

gpu
Mean prediction time for batch size 1 is: 5171.07 microseconds (total time 0.52 seconds)
Mean prediction time for batch size 2 is: 2775.60 microseconds (total time 0.56 seconds)
Mean prediction time for batch size 4 is: 1584.09 microseconds (total time 0.63 seconds)
Mean prediction time for batch size 8 is: 892.08 microseconds (total time 0.71 seconds)
Mean prediction time for batch size 16 is: 375.71 microseconds (total time 0.60 seconds)
Mean prediction time for batch size 32 is: 234.62 microseconds (total time 0.75 seconds)
Mean prediction time for batch size 64 is: 111.04 microseconds (total time 0.71 seconds)
Mean prediction time for batch size 128 is: 55.80 microseconds (total time 0.71 seconds)
Mean prediction time for batch size 256 is: 31.70 microseconds (total time 0.81 seconds)
Mean prediction time for batch size 512 is: 21.08 microseconds (total time 1.08 seconds)
Mean prediction time for batch size 1024 is: 11.83 microseconds (total time 1.21 seconds)
Mean prediction time for batch size 2048 is: 8.63 microseconds (total time 1.77 seconds)
Mean prediction time for batch size 4096 is: 6.41 microseconds (total time 2.62 seconds)
Mean prediction time for batch size 8192 is: 7.13 microseconds (total time 5.84 seconds)
"""