import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices("GPU")
print(f"GPU Devices: {gpu_devices}")
if gpu_devices:
    print("TensorFlow Metal GPU acceleration IS available.")
    # You can also check details of the GPU
    for gpu in gpu_devices:
        print(f"  Details: {tf.config.experimental.get_device_details(gpu)}")
else:
    print(
        "TensorFlow Metal GPU acceleration IS NOT available. (This is unexpected if tensorflow-metal installed correctly)"
    )
