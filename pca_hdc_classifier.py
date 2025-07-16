import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torchhd
from torchhd import embeddings, functional
from sklearn.decomposition import PCA

from binhd.embeddings import ScatterCode
from binhd.classifiers import BinHD

# Configurações
model_path = 'vehicle_color_classifier_resnet50.pth'
data_train_dir = 'data'
data_test_dir = 'data_test'
batch_size = 64
hd_dim = 1000
num_levels = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Transformacoes para as imagens
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregar datasets
train_dataset = datasets.ImageFolder(data_train_dir, transform=transform)
test_dataset = datasets.ImageFolder(data_test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

# Carregar ResNet50 para extração de features
model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Extrator de features (sem a camada fc)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# Determinar min e max a partir de uma amostra do dataset
sample_feats = []
with torch.no_grad():
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        feats = feature_extractor(inputs)
        feats = torch.flatten(feats, 1).cpu()
        sample_feats.append(feats)
        if len(sample_feats) > 2:
            break
sample_feats = torch.cat(sample_feats, dim=0)

# Aplicar PCA
print("📉 Reduzindo dimensionalidade com PCA...")
pca = PCA(n_components=128)
sample_feats_reduced = pca.fit_transform(sample_feats)
min_val, max_val = sample_feats_reduced.min(), sample_feats_reduced.max()

# Codificacao HDC com record-based encoding
class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__()
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.uint8)
        self.value = ScatterCode(levels, out_features, low=low, high=high)

    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

record_encoder = RecordEncoder(hd_dim, 128, num_levels, min_val, max_val).to(device)

# Instanciar classificador BinHD
hdc_model = BinHD(
    n_dimensions=1000,
    n_classes=num_classes,
    device=device
)
hdc_model.binary = True
hdc_model.bipolar = False

# Treinamento em batches
print("\n🚂 Treinando em batches...")
record_encoder.eval()
feature_extractor.eval()

X_train_batches = []
y_train_batches = []

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"🔁 Processando batch {batch_idx + 1}...")
        inputs = inputs.to(device)
        feats = feature_extractor(inputs)
        feats = torch.flatten(feats, 1).cpu()
        feats = torch.from_numpy(pca.transform(feats))
        encoded_feats = record_encoder(feats.to(device))

        X_train_batches.append(encoded_feats)
        y_train_batches.append(labels)

X_train_tensor = torch.tensor(np.concatenate(X_train_batches), dtype=torch.int8)
y_train_tensor = torch.cat(y_train_batches, dim=0)

print("🧠 Treinando classificador HDC...")
hdc_model.fit(X_train_tensor, y_train_tensor)


# Predição em batches
y_true, y_pred = [], []
print("\n🔍 Predizendo em batches...")
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        print(f"🔮 Predizendo batch {batch_idx + 1}...")
        inputs = inputs.to(device)
        feats = feature_extractor(inputs)
        feats = torch.flatten(feats, 1).cpu()
        feats = torch.from_numpy(pca.transform(feats))
        encoded_feats = record_encoder(feats.to(device))

        preds = hdc_model.predict(encoded_feats)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

# Avaliação
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusão - HDC por batches")
plt.tight_layout()
plt.show()
