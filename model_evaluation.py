import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input  # ou outro preprocess

# Caminho do modelo salvo
model_path = 'vehicle_color_classifier_resnet50.h5'
test_dir = 'data_test'

# Carrega o modelo
model = load_model(model_path)

# Lista de classes (nomes das pastas dentro de data_test)
# with open("class_names.json") as f:
#     class_names = json.load(f)

class_names = sorted(os.listdir('data'))
print(class_names)

y_true = []
y_pred = []

# Percorre cada classe (pasta de cor)
for label in class_names:
    class_path = os.path.join(test_dir, label)
    for fname in os.listdir(class_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, fname)

        # Carrega imagem e pré-processa
        img = image.load_img(img_path, target_size=(224, 224))  # ajuste ao tamanho do seu modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predição
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Registra resultado
        y_true.append(label)
        y_pred.append(predicted_class)

# Gera a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()
