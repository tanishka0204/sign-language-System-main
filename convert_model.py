import tensorflow as tf

KERAS_MODEL = "model/keypoint_classifier/keypoint_classifier.keras"
TFLITE_MODEL = "model/keypoint_classifier/keypoint_classifier.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

print("Model loaded.")
model.summary()

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Do NOT use the newer experimental quantization settings.
# Start with a normal float32 TFLite model for maximum compatibility.
tflite_model = converter.convert()

with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print("\nSaved:", TFLITE_MODEL)
print("Size:", len(tflite_model), "bytes")

print("\nTesting the generated model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

print("SUCCESS!")
print("Input:", interpreter.get_input_details())
print("Output:", interpreter.get_output_details())