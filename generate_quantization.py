import tensorflow as tf

model = tf.keras.models.load_model("vehicle_color_classifier_resnet50_savedmodel")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_model = converter.convert()

# Salva modelo quantizado
with open('model_quantized.tflite', 'wb') as f:
    f.write(quantized_model)
