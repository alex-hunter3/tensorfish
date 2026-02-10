import os

# disables warnings from tensorfeed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("models/tensorfish.keras")

spec = (tf.TensorSpec((None, 8, 8, 19), tf.float32, name="input"),)

# Convert and save
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("models/tensorfish.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())