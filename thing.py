import tensorflow as tf
import tf2onnx
import onnx

plant_model = tf.keras.models.load_model('./static/models/threeclassplant.keras')
plant_model.output_names=['output']
input_signature = [tf.TensorSpec([None, 384, 384, 3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(plant_model, input_signature, opset=13)
onnx.save(onnx_model, "plantnew.onnx")
