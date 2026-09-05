import tensorflow as tf
import numpy as np

KERAS_MODEL = "model/keypoint_classifier/keypoint_classifier.keras"
TFLITE_MODEL = "model/keypoint_classifier/keypoint_classifier.tflite"

print("Loading original model...")
original = tf.keras.models.load_model(KERAS_MODEL, compile=False)

print("Original model loaded.")
original.summary()

# ---------------------------------------------------------
# Create inference model WITHOUT BatchNormalization/Dropout
# ---------------------------------------------------------

bn = original.get_layer("batch_normalization")
dense0 = original.get_layer("dense")
dense1 = original.get_layer("dense_1")
dense2 = original.get_layer("dense_2")
dense3 = original.get_layer("dense_3")

# BatchNorm parameters
gamma, beta, moving_mean, moving_var = bn.get_weights()
epsilon = bn.epsilon

# Fold BatchNorm into first Dense layer.
#
# BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
#
# Dense(BN(x)) can be represented as:
#
# Dense(x, W_new, b_new)
#
# where:
# W_new = W * gamma / sqrt(var + eps)
# b_new = b + beta_effect
#
# The Dense layer in this model follows BatchNorm.

W, b = dense0.get_weights()

scale = gamma / np.sqrt(moving_var + epsilon)

W_new = W * scale[:, np.newaxis]
b_new = b - np.sum(W * (scale * moving_mean)[:, np.newaxis], axis=0) + beta @ W

# ---------------------------------------------------------
# Rebuild model without BN and Dropout
# ---------------------------------------------------------

inputs = tf.keras.Input(shape=(42,), name="input")

x = tf.keras.layers.Dense(
    128,
    activation=dense0.activation,
    name="dense"
)(inputs)

x = tf.keras.layers.Dense(
    64,
    activation=dense1.activation,
    name="dense_1"
)(x)

x = tf.keras.layers.Dense(
    32,
    activation=dense2.activation,
    name="dense_2"
)(x)

outputs = tf.keras.layers.Dense(
    26,
    activation=dense3.activation,
    name="dense_3"
)(x)

model = tf.keras.Model(inputs, outputs)

# Set folded first Dense weights
model.get_layer("dense").set_weights([W_new, b_new])

# Copy remaining weights
model.get_layer("dense_1").set_weights(dense1.get_weights())
model.get_layer("dense_2").set_weights(dense2.get_weights())
model.get_layer("dense_3").set_weights(dense3.get_weights())

print("\nNew inference model:")
model.summary()

# ---------------------------------------------------------
# Compare original and folded model
# ---------------------------------------------------------

test_input = np.random.random((1, 42)).astype(np.float32)

original_prediction = original(test_input, training=False).numpy()
new_prediction = model(test_input, training=False).numpy()

difference = np.max(
    np.abs(original_prediction - new_prediction)
)

print("\nMaximum prediction difference:", difference)

# ---------------------------------------------------------
# Convert to TFLite
# ---------------------------------------------------------

print("\nConverting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print("Saved:", TFLITE_MODEL)
print("Size:", len(tflite_model), "bytes")

# ---------------------------------------------------------
# Verify TFLite model
# ---------------------------------------------------------

print("\nTesting TFLite model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)

interpreter.set_tensor(
    input_details[0]["index"],
    test_input
)

interpreter.invoke()

tflite_prediction = interpreter.get_tensor(
    output_details[0]["index"]
)

difference_tflite = np.max(
    np.abs(original_prediction - tflite_prediction)
)

print("TFLite prediction difference:", difference_tflite)

print("\n===================================")
print("SUCCESS: TFLite model is working!")
print("===================================")