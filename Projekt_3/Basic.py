import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Parametry uczenia
BATCH_SIZE = 8
EPOCHS = 7
LEARNING_RATE = 3e-4
DATA_DIR = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Definicja własnego szumu
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_data_loaders(data_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    classes = dataset.classes
    print(f'Znalezione klasy: {classes}')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = DEVICE.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, classes, dataset


def initialize_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model.to(DEVICE)


def get_class_weights(dataset):
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets)
    weights = 1.0 / (class_counts.float() + 1e-8)
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(DEVICE)


def plot_training_history(history, epochs):
    """Funkcja do rysowania wykresów z przebiegu treningu"""
    print("\nGenerowanie wykresów treningu...")
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Wykres 1: Strata (Loss)
    ax1.plot(epochs_range, history['train_loss'], label='Trening', marker='o')
    ax1.plot(epochs_range, history['val_loss'], label='Walidacja', marker='o')
    ax1.set_title('Wykres Straty (Loss)')
    ax1.set_xlabel('Epoki')
    ax1.set_ylabel('Strata')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Wykres 2: Dokładność (Accuracy)
    ax2.plot(epochs_range, history['train_acc'], label='Trening', marker='o', color='green')
    ax2.plot(epochs_range, history['val_acc'], label='Walidacja', marker='o', color='red')
    ax2.set_title('Wykres Dokładności (Accuracy)')
    ax2.set_xlabel('Epoki')
    ax2.set_ylabel('Dokładność [%]')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Wykres 3: Learning Rate
    ax3.plot(epochs_range, history['lr'], label='Learning Rate', marker='o', color='purple')
    ax3.set_title('Krzywa Szybkości Uczenia (LR)')
    ax3.set_xlabel('Epoki')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('historia_gitar.png', dpi=300)
    print("Zapisano wykresy jako 'historia_gitar.png'")
    plt.close()

def main():
    print('Wczytywanie datasetu...')
    train_loader, val_loader, classes, dataset = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = initialize_model(len(classes))
    print(f'Używane urządzenie: {DEVICE}')

    class_weights = get_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-3
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    best_model_path = 'best_guitar_classifier.pth'

    print('\nRozpoczynamy trening...')
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = 100.0 * correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Nowy najlepszy model zapisany: {best_val_acc:.2f}%')

        print(
            f'Epoka [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | '
            f'Strata T/W: {train_loss:.4f}/{val_loss:.4f} | '
            f'Dokładność T/W: {train_acc:.2f}% / {val_acc:.2f}%'
        )

    print('\nTrening zakończony pomyślnie!')
    torch.save(model.state_dict(), 'guitar_classifier.pth')
    print("Zapisano model jako 'guitar_classifier.pth'")
    print(f'Najlepsza walidacyjna dokładność: {best_val_acc:.2f}%')

    plot_training_history(history, EPOCHS)


if __name__ == '__main__':
    main()