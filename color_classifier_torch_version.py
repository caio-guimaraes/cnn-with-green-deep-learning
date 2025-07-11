import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import time
import os


def main():
    # Diretório das imagens (subpastas por cor)
    data_dir = 'data'
    img_size = (224, 224)
    batch_size = 32
    seed = 42

    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)


    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
    }

    # Carrega datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir),
                                transform=data_transforms[x])
        for x in ['train', 'val']
    }

    dataset_size = len(image_datasets['train'])
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(image_datasets['train'], [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # Salva as classes ANTES de aplicar map()
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Carrega a ResNet50 pré-treinada
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    # Substitui a última camada totalmente conectada
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes)
    )

    model = model.to(device)

    # Otimizador e loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    # Treinamento
    print("\nIniciando treinamento...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 30)
        epoch_start = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}", unit="batch"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            elapsed = time.time() - epoch_start

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - Tempo: {elapsed:.2f}s")

    print("✅ Treinamento concluído.")

    # Salva o modelo
    torch.save(model.state_dict(), "vehicle_color_classifier_resnet50.pth")
    print("✅ Modelo salvo como vehicle_color_classifier_resnet50.pth")

if __name__ == "__main__":
    main()