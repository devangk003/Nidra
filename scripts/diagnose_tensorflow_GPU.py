import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))

# Try to create a tensor on GPU
try:
    with tf.device('/GPU:0'):
        test_tensor = tf.constant([1.0, 2.0, 3.0])
    print("TensorFlow is using GPU.")
except:
    print("TensorFlow is NOT using GPU.")
