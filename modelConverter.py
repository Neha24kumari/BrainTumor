from keras.models import load_model
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.FATAL)


#model=load_model('BrainTumor10Epochs.keras')

#converter=tf.lite.TFLiteConverter.from_keras_model(model)

#tf_lite_model=converter.convert()

#with open('model.tflite', 'wb') as f:
 #   f.write(tf_lite_model)

model = load_model('BrainTumor10Epochs.keras')

# Try TensorFlow Lite conversion with optimizations and quantization

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TensorFlow Lite format
tf_lite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tf_lite_model)

print("TensorFlow Lite conversion successful.")