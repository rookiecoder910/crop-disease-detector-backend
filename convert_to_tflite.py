import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HDF5_PATH = os.path.join(BASE_DIR, "model", "Model.hdf5")
TFLITE_PATH = os.path.join(BASE_DIR, "model", "model_origin.tflite")

print("Loading HDF5 model...")
model = tf.keras.models.load_model(HDF5_PATH, compile=False)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)


tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(" model.tflite created successfully")
print("Saved at:", TFLITE_PATH)
