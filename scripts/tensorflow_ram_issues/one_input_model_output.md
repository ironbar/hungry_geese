# One input model output

## GPU

```bash
2021-03-15 15:43:30,329 - __main__ - INFO - Model created
2021-03-15 15:43:30,329 - __main__ - INFO - Ram usage: 1.6 GB   available: 60.7 GB
2021-03-15 15:43:44,412 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 15:43:44,413 - __main__ - INFO - Data created
2021-03-15 15:43:44,413 - __main__ - INFO - Ram usage: 9.6 GB   available: 52.6 GB
2021-03-15 15:43:44.414740: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 15:43:47.693161: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 15:43:47.710656: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
Epoch 1/3
2021-03-15 15:43:47.930387: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 15:43:48.431917: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
1236/1236 [==============================] - 5s 3ms/step - loss: 0.2042
2021-03-15 15:43:52,680 - __main__ - INFO - Ram usage: 19.2 GB  available: 43.2 GB
Epoch 2/3
1236/1236 [==============================] - 4s 3ms/step - loss: 0.0838
2021-03-15 15:43:56,988 - __main__ - INFO - Ram usage: 19.1 GB  available: 43.2 GB
Epoch 3/3
1236/1236 [==============================] - 4s 3ms/step - loss: 0.0838
2021-03-15 15:44:01,275 - __main__ - INFO - Ram usage: 19.0 GB  available: 43.3 GB
```

RAM usage is double than the dataset size

## CPU

```bash
2021-03-15 15:44:14.518235: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 15:44:14,530 - __main__ - INFO - Model created
2021-03-15 15:44:14,530 - __main__ - INFO - Ram usage: 0.3 GB   available: 61.9 GB
2021-03-15 15:44:28,780 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 15:44:28,780 - __main__ - INFO - Data created
2021-03-15 15:44:28,780 - __main__ - INFO - Ram usage: 8.3 GB   available: 53.8 GB
2021-03-15 15:44:28.782215: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 15:44:32.054516: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 15:44:32.074717: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
Epoch 1/3
1236/1236 [==============================] - 5s 3ms/step - loss: 0.1999
2021-03-15 15:44:36,789 - __main__ - INFO - Ram usage: 17.0 GB  available: 45.1 GB
Epoch 2/3
1236/1236 [==============================] - 4s 3ms/step - loss: 0.0838
2021-03-15 15:44:40,898 - __main__ - INFO - Ram usage: 16.9 GB  available: 45.2 GB
Epoch 3/3
1236/1236 [==============================] - 4s 3ms/step - loss: 0.0837
2021-03-15 15:44:45,080 - __main__ - INFO - Ram usage: 16.8 GB  available: 45.3 GB
```

When using cpu the memory also doubles, however RAM usage is slower.

## CPU no eager execution

```bash
(goose) gbarbadillo@africanus:/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese$ python scripts/tensorflow_ram_issues/one_input_model.py 
2021-03-15 15:50:11.092477: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 15:50:11,899 - __main__ - INFO - Model created
2021-03-15 15:50:11,899 - __main__ - INFO - Ram usage: 0.3 GB   available: 61.6 GB
2021-03-15 15:50:26,018 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 15:50:26,018 - __main__ - INFO - Data created
2021-03-15 15:50:26,018 - __main__ - INFO - Ram usage: 8.3 GB   available: 53.6 GB
Train on 10000000 samples
Epoch 1/3
2021-03-15 15:50:26.416318: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 15:50:26.417075: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-03-15 15:50:26.450813: E tensorflow/stream_executor/cuda/cuda_driver.cc:328] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-03-15 15:50:26.450858: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: africanus
2021-03-15 15:50:26.450867: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: africanus
2021-03-15 15:50:26.450976: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:200] libcuda reported version is: 460.39.0
2021-03-15 15:50:26.451007: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:204] kernel reported version is: 460.39.0
2021-03-15 15:50:26.451017: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:310] kernel version seems to match DSO: 460.39.0
2021-03-15 15:50:26.451462: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-03-15 15:50:26.452379: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 15:50:26.460978: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-03-15 15:50:26.463162: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
 9949984/10000000 [============================>.] - ETA: 0s - loss: 0.14382021-03-15 15:50:35,526 - __main__ - INFO - Ram usage: 8.5 GB        available: 53.5 GB
10000000/10000000 [==============================] - 9s 1us/sample - loss: 0.1435
Epoch 2/3
 9966176/10000000 [============================>.] - ETA: 0s - loss: 0.08392021-03-15 15:50:44,957 - __main__ - INFO - Ram usage: 8.5 GB        available: 53.5 GB
10000000/10000000 [==============================] - 9s 1us/sample - loss: 0.0839
Epoch 3/3
 9949984/10000000 [============================>.] - ETA: 0s - loss: 0.08372021-03-15 15:50:54,526 - __main__ - INFO - Ram usage: 8.5 GB        available: 53.5 GB
10000000/10000000 [==============================] - 10s 1us/sample - loss: 0.0837
```

In this case the memory is not doubled.

## GPU no eager execution

```bash
(goose) gbarbadillo@africanus:/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese$ python scripts/tensorflow_ram_issues/one_input_model.py 
2021-03-15 15:52:13.084075: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 15:52:13,912 - __main__ - INFO - Model created
2021-03-15 15:52:13,913 - __main__ - INFO - Ram usage: 0.3 GB   available: 61.7 GB
2021-03-15 15:52:28,183 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 15:52:28,183 - __main__ - INFO - Data created
2021-03-15 15:52:28,183 - __main__ - INFO - Ram usage: 8.3 GB   available: 53.6 GB
Train on 10000000 samples
Epoch 1/3
2021-03-15 15:52:28.579359: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 15:52:28.580108: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-03-15 15:52:28.620406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 15:52:28.620443: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 15:52:28.622195: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 15:52:28.622258: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 15:52:28.622990: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 15:52:28.623182: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 15:52:28.624994: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 15:52:28.625438: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 15:52:28.625518: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 15:52:28.627265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 15:52:28.627622: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-03-15 15:52:28.628271: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 15:52:28.629190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 15:52:28.629211: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 15:52:28.629226: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 15:52:28.629234: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 15:52:28.629241: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 15:52:28.629249: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 15:52:28.629256: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 15:52:28.629264: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 15:52:28.629271: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 15:52:28.631054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 15:52:28.631076: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 15:52:29.015296: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-03-15 15:52:29.015330: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2021-03-15 15:52:29.015335: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2021-03-15 15:52:29.017946: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 21825 MB memory) -> physical GPU (device: 0, name: GeForce RTX 3090, pci bus id: 0000:65:00.0, compute capability: 8.6)
2021-03-15 15:52:29.023329: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-03-15 15:52:29.025454: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
2021-03-15 15:52:29.275144: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 15:52:29.737960: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
10000000/10000000 [==============================] - ETA: 0s - loss: 0.09462021-03-15 15:52:38,531 - __main__ - INFO - Ram usage: 10.5 GB       available: 51.6 GB
10000000/10000000 [==============================] - 10s 1us/sample - loss: 0.0946
Epoch 2/3
 9949984/10000000 [============================>.] - ETA: 0s - loss: 0.08362021-03-15 15:52:47,932 - __main__ - INFO - Ram usage: 10.5 GB       available: 51.6 GB
10000000/10000000 [==============================] - 9s 1us/sample - loss: 0.0836
Epoch 3/3
 9982368/10000000 [============================>.] - ETA: 0s - loss: 0.08362021-03-15 15:52:57,232 - __main__ - INFO - Ram usage: 10.5 GB       available: 51.6 GB
10000000/10000000 [==============================] - 9s 1us/sample - loss: 0.0836
```

Half the RAM is used, but epoch time is doubled.

## GPU but using tf.data instead of numpy

```bash
(goose) gbarbadillo@africanus:/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese$ python scripts/tensorflow_ram_issues/one_input_model.py 
2021-03-15 16:05:17.446765: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:05:18.269126: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 16:05:18.269835: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-03-15 16:05:18.309823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 16:05:18.309857: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:05:18.311662: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:05:18.311721: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 16:05:18.312450: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 16:05:18.312628: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 16:05:18.314489: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 16:05:18.314906: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 16:05:18.314983: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 16:05:18.316751: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 16:05:18.317029: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-03-15 16:05:18.317630: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 16:05:18.318560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 16:05:18.318580: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:05:18.318596: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:05:18.318604: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 16:05:18.318613: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 16:05:18.318621: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 16:05:18.318630: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 16:05:18.318638: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 16:05:18.318647: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 16:05:18.320234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 16:05:18.320256: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:05:18.691321: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-03-15 16:05:18.691353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2021-03-15 16:05:18.691360: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2021-03-15 16:05:18.693754: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 21763 MB memory) -> physical GPU (device: 0, name: GeForce RTX 3090, pci bus id: 0000:65:00.0, compute capability: 8.6)
2021-03-15 16:05:18,886 - __main__ - INFO - Model created
2021-03-15 16:05:18,886 - __main__ - INFO - Ram usage: 1.6 GB   available: 60.1 GB
2021-03-15 16:05:33,351 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 16:05:33.352254: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 16:05:36,665 - __main__ - INFO - Data created
2021-03-15 16:05:36,666 - __main__ - INFO - Ram usage: 17.7 GB  available: 44.0 GB
2021-03-15 16:05:36.690145: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 16:05:40.635858: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 1/3
2021-03-15 16:05:44.742668: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 16:05:44.762636: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
2021-03-15 16:05:44.806411: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:05:45.311753: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
1236/1236 [==============================] - 18s 14ms/step - loss: 0.1995
2021-03-15 16:06:02,853 - __main__ - INFO - Ram usage: 10.4 GB  available: 51.4 GB
2021-03-15 16:06:02.854584: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 2/3
1236/1236 [==============================] - 17s 14ms/step - loss: 0.0838
2021-03-15 16:06:24,237 - __main__ - INFO - Ram usage: 10.4 GB  available: 51.3 GB
2021-03-15 16:06:24.239796: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 3/3
1236/1236 [==============================] - 18s 15ms/step - loss: 0.0837
2021-03-15 16:06:46,229 - __main__ - INFO - Ram usage: 10.4 GB  available: 51.3 GB
```

Good ram usage but slow speed.

##  GPU but using tf.data instead of numpy, prefetch

```bash
(goose) gbarbadillo@africanus:/mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese$ python scripts/tensorflow_ram_issues/one_input_model.py 
2021-03-15 16:12:23.929928: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:12:24.752441: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 16:12:24.753187: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-03-15 16:12:24.790029: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 16:12:24.790075: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:12:24.792369: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:12:24.792445: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 16:12:24.793256: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 16:12:24.793443: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 16:12:24.795252: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 16:12:24.795676: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 16:12:24.795755: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 16:12:24.797544: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 16:12:24.797829: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-03-15 16:12:24.798617: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-15 16:12:24.799543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: GeForce RTX 3090 computeCapability: 8.6
coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
2021-03-15 16:12:24.799562: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:12:24.799577: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:12:24.799585: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-03-15 16:12:24.799592: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-03-15 16:12:24.799599: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-03-15 16:12:24.799607: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-03-15 16:12:24.799616: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-03-15 16:12:24.799627: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-03-15 16:12:24.801275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-03-15 16:12:24.801299: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-03-15 16:12:25.172890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-03-15 16:12:25.172922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2021-03-15 16:12:25.172927: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2021-03-15 16:12:25.175589: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 21838 MB memory) -> physical GPU (device: 0, name: GeForce RTX 3090, pci bus id: 0000:65:00.0, compute capability: 8.6)
2021-03-15 16:12:25,366 - __main__ - INFO - Model created
2021-03-15 16:12:25,366 - __main__ - INFO - Ram usage: 1.6 GB   available: 59.9 GB
2021-03-15 16:12:39,992 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 16:12:39.993366: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 16:12:43,381 - __main__ - INFO - Data created
2021-03-15 16:12:43,381 - __main__ - INFO - Ram usage: 17.7 GB  available: 43.8 GB
2021-03-15 16:12:43.406933: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 16:12:47.389808: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 1/3
2021-03-15 16:12:51.505723: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 16:12:51.526534: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
2021-03-15 16:12:51.555140: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:12:52.051874: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
1236/1236 [==============================] - 13s 10ms/step - loss: 0.2037
2021-03-15 16:13:04,008 - __main__ - INFO - Ram usage: 10.5 GB  available: 51.1 GB
2021-03-15 16:13:04.010918: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 2/3
1236/1236 [==============================] - 12s 9ms/step - loss: 0.0841
2021-03-15 16:13:19,776 - __main__ - INFO - Ram usage: 10.5 GB  available: 51.1 GB
2021-03-15 16:13:19.779680: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
Epoch 3/3
1236/1236 [==============================] - 11s 9ms/step - loss: 0.0837
2021-03-15 16:13:35,070 - __main__ - INFO - Ram usage: 10.5 GB  available: 51.2 GB
```

## GPU without shuffle on fit method

```bash
0, compute capability: 8.6)
2021-03-15 16:33:52,052 - __main__ - INFO - Model created
2021-03-15 16:33:52,052 - __main__ - INFO - Ram usage: 1.6 GB   available: 59.5 GB
2021-03-15 16:34:06,265 - __main__ - INFO - Input size: 8.0 GB
2021-03-15 16:34:06,265 - __main__ - INFO - Data created
2021-03-15 16:34:06,265 - __main__ - INFO - Ram usage: 9.6 GB   available: 51.4 GB
2021-03-15 16:34:06.266943: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 8000000000 exceeds 10% of free system memory.
2021-03-15 16:34:09.553584: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-15 16:34:09.570625: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3699850000 Hz
Epoch 1/3
2021-03-15 16:34:09.792518: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-03-15 16:34:10.272362: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
1236/1236 [==============================] - 5s 4ms/step - loss: 0.2112
2021-03-15 16:34:14,663 - __main__ - INFO - Ram usage: 19.2 GB  available: 42.0 GB
Epoch 2/3
1236/1236 [==============================] - 5s 4ms/step - loss: 0.0839
2021-03-15 16:34:19,175 - __main__ - INFO - Ram usage: 19.1 GB  available: 42.0 GB
Epoch 3/3
1236/1236 [==============================] - 4s 4ms/step - loss: 0.0837
2021-03-15 16:34:23,541 - __main__ - INFO - Ram usage: 19.0 GB  available: 42.1 GB
```

## Learnings

It seems that is creating an internal tf.dataset and that's what creates the double size in RAM memory.
However when I create the tf.dataset myself it is much slower.

Can I recreate the problem in colab?
No, the problem does not manifest on colab

If using a generator the problem does not happen.

Disabling eager execution solves the RAM problem, but at the cost of being much slower.

https://github.com/tensorflow/tensorflow/issues/31312

The problem remains unsolved, I suggest to use a generator to feed the data and avoid memory problems.

## Ideas

- Tf.data
- Eager execution

```bash
```

