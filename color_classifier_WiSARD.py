import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer
import time
import os

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

        # Codificação com Termômetro
        bits_encoding = 20
        encoding = Thermometer(bits_encoding).fit(X)
        X_bin = encoding.binarize(X).flatten(start_dim=1)

        return X_bin, y

    print("🔍 Extraindo features de treino...")
    X_train, y_train = extract_features(train_loader)
    print("✅ Treino:", X_train.shape)

    print("🔍 Extraindo features de teste...")
    X_test, y_test = extract_features(test_loader)
    print("✅ Teste:", X_test.shape)

    # WiSARD
    entry_size = X_train.shape[1]
    tuple_size = 8

    with torch.no_grad():
        wisard = Wisard(entry_size, num_classes, tuple_size, bleaching=True)
        wisard.fit(X_train, y_train)

    # Avaliação
    y_pred = wisard.predict(X_test)

    # Classification report
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=train_dataset.classes))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Matriz de Confusão WiSARD")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()