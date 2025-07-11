import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite", num_threads=8)
interpreter.allocate_tensors()

img_size = (224, 224)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_test",
    image_size=img_size,
    batch_size=1,
    shuffle=False
)

class_names = test_ds.class_names

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_true = []
y_pred = []

for image_batch, label_batch in tqdm(test_ds):
    image = image_batch.numpy()

    # (Se o modelo usa quantização, adapte para uint8)
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        image = image / input_scale + input_zero_point
        image = image.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output[0])
    y_pred.append(predicted_class)
    y_true.append(label_batch.numpy()[0])


print("\n🎯 Relatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()
