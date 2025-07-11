import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Caminhos
model_path = 'vehicle_color_classifier_resnet50.pth'
test_dir = 'data_test'

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Lista de classes (nomes das pastas dentro de data_test)
class_names = sorted(os.listdir(test_dir))
num_classes = len(class_names)

# Modelo
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transforms iguais ao treino
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

y_true = []
y_pred = []

# Lista de todas as imagens e rótulos
samples = []
for label in class_names:
    class_path = os.path.join(test_dir, label)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            samples.append((label, os.path.join(class_path, fname)))

# Classificação
start_time = time.time()
for label, img_path in tqdm(samples, desc="Validando", unit="img"):
    # Carrega imagem e pré-processa
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predição
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        predicted_class = class_names[pred_idx.item()]

    # Registra resultado
    y_true.append(label)
    y_pred.append(predicted_class)

elapsed = time.time() - start_time
print(f"\n✅ Classificação concluída em {elapsed:.2f} s para {len(samples)} imagens.")

# Encoding
le = LabelEncoder()
le.fit(class_names)
y_true_enc = le.transform(y_true)
y_pred_enc = le.transform(y_pred)

# Matriz de confusão
cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(class_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(
    y_true_enc,
    y_pred_enc,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0
)
print("\n📊 Classification Report:\n")
print(report)