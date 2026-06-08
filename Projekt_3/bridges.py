import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image

# Parametry uczenia
BATCH_SIZE = 16 
EPOCHS = 15 # Zwiększamy ilość epok, bo używamy Dropoutu i fine-tuningu
LEARNING_RATE = 5e-4 # Bazowy learning rate
DATA_DIR = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALLOWED_CLASSES = ['Floyd_sytle', 'Hardtail_style', 'Tremolo_style', 'Tune_o_matic_style']

# 1. Szum Gaussa
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# 2. Bezpieczny Custom Dataset
class GuitarBridgesDataset(Dataset):
    def __init__(self, root_dir, allowed_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = allowed_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(allowed_classes)}
        
        self.image_paths = []
        self.targets = [] 

        for cls_name in allowed_classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Ostrzeżenie: Nie znaleziono folderu: {cls_dir}")
                continue

            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.targets.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def get_data_loaders(data_dir, batch_size):
    # Znacznie silniejsza augmentacja zapobiegająca przeuczeniu
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(600, scale=(0.7, 1.0)), # Zapobiega gubieniu całego mostka
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.03),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(650),
        transforms.CenterCrop(600),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset_train = GuitarBridgesDataset(root_dir=data_dir, allowed_classes=ALLOWED_CLASSES, transform=train_transforms)
    full_dataset_val = GuitarBridgesDataset(root_dir=data_dir, allowed_classes=ALLOWED_CLASSES, transform=val_transforms)
    
    classes = full_dataset_train.classes
    print(f'Znalezione klasy: {classes}')

    train_size = int(0.8 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset_train, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(full_dataset_val, [train_size, val_size], generator=generator)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = DEVICE.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, classes, full_dataset_train


def initialize_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ODBLOKOWANIE LAYER4 DO FINE-TUNINGU (Kluczowa zmiana do dokładności!)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # PODWÓJNY DROPOUT (Zabijanie przeuczenia)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    return model.to(DEVICE)


def get_class_weights(dataset):
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets)
    weights = 1.0 / (class_counts.float() + 1e-8)
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(DEVICE)


def plot_training_history(history, epochs):
    print("\nGenerowanie wykresów treningu...")
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(epochs_range, history['train_loss'], label='Trening', marker='o')
    ax1.plot(epochs_range, history['val_loss'], label='Walidacja', marker='o')
    ax1.set_title('Wykres Straty (Loss)')
    ax1.set_xlabel('Epoki')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(epochs_range, history['train_acc'], label='Trening', marker='o', color='green')
    ax2.plot(epochs_range, history['val_acc'], label='Walidacja', marker='o', color='red')
    ax2.set_title('Dokładność (Accuracy)')
    ax2.set_xlabel('Epoki')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax3.plot(epochs_range, history['lr'], label='Learning Rate', marker='o', color='purple')
    ax3.set_title('Krzywa LR')
    ax3.set_xlabel('Epoki')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('historia_mostki.png', dpi=300)
    plt.close()

def main():
    print('Wczytywanie datasetu...')
    train_loader, val_loader, classes, dataset = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = initialize_model(len(classes))
    print(f'Używane urządzenie: {DEVICE}')

    class_weights = get_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # DISCRIMINATIVE LEARNING RATES: 
    # Stare warstwy (layer4) uczą się 10x wolniej, nowe (fc) uczą się normalnie
    optimizer = optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-3)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    best_model_path = 'best_bridge_classifier.pth'

    print('\nRozpoczynamy zoptymalizowany trening...')
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
        current_lr = optimizer.param_groups[1]['lr'] # Pobieramy LR dla warstwy fc

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        scheduler.step()

        # Automatyczny zapis najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'⭐ Nowy najlepszy model zapisany: {best_val_acc:.2f}%')

        print(
            f'Epoka [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | '
            f'Strata T/W: {train_loss:.4f}/{val_loss:.4f} | '
            f'Dokładność T/W: {train_acc:.2f}% / {val_acc:.2f}%'
        )

    print('\nTrening zakończony pomyślnie!')
    print(f'🏆 Najlepsza wyciągnięta dokładność walidacyjna: {best_val_acc:.2f}%')
    print("Do predykcji pamiętaj używać pliku 'best_bridge_classifier.pth'")

    plot_training_history(history, EPOCHS)

if __name__ == '__main__':
    main()