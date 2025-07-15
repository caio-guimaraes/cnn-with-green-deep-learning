import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torchmetrics

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__()
        self.position = embeddings.Random(size, out_features)
        self.value = embeddings.Level(levels, out_features, low=low, high=high)
    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

def main():
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

    # Extrator de features (sem o otimizador e loss)
    feature_extractor = model
    # feature_extractor = nn.Sequential(*list(model.children())[:-1])
    # feature_extractor.to(device)
    # feature_extractor.eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("data", transform=transform)
    test_dataset  = datasets.ImageFolder("data_test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    #  Extrair features + termômetro
    def extract_features(dataloader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                feats = feature_extractor(inputs)
                feats = torch.flatten(feats, 1)

                all_features.append(feats.cpu())
                all_labels.extend(labels.cpu())

        # Junta todos os batches em um único tensor
        X = torch.cat(all_features, dim=0) 
        y = torch.tensor(all_labels)

        return X, y

    print("🔍 Extraindo features de treino...")
    X_train, y_train = extract_features(train_loader)
    print("✅ Treino X:", X_train.shape)
    print("✅ Treino y:", y_train.shape)

    print("🔍 Extraindo features de teste...")
    X_test, y_test = extract_features(test_loader)
    print("✅ Teste X:", X_test.shape)
    print("✅ Teste y:", y_test.shape)

    # HDC
    minGlobal = X_train.min().item()
    maxGlobal = X_train.max().item()
    
    print(minGlobal)
    print(maxGlobal)

    num_features = X_train.shape[1]

    DIMENSIONS = 10000
    NUM_LEVELS = int(maxGlobal - minGlobal + 1)  # por ser categórico, pode usar isso

    record_encode = RecordEncoder(DIMENSIONS, num_features, NUM_LEVELS, minGlobal, maxGlobal)
    record_encode = record_encode.to(device)

    hd_model = Centroid(DIMENSIONS, num_classes)
    hd_model = hd_model.to(device)

    # Treinamento
    with torch.no_grad():
        samples = X_train.to(device)
        labels = y_train.to(device)

        samples_hv = record_encode(samples)
        hd_model.add(samples_hv, labels)
        hd_model.weight.data = torch.where(hd_model.weight > 0, 1, -1)

    # Previsão
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        test_samples = X_test.to(device)
        test_labels = y_test.to(device)

        test_samples_hv = record_encode(test_samples)
        outputs = hd_model(test_samples_hv)
        predictions = torch.argmax(outputs, dim=1)

    # Classification report
    print("\n📊 Classification Report:\n")
    print(classification_report(test_labels, predictions, target_names=train_dataset.classes))

    # Matriz de confusão
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Matriz de Confusão WiSARD")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()