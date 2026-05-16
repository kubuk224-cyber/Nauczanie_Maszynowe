import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Parametry uczenia
BATCH_SIZE = 32
EPOCHS = 30 # Zwiększyłem do 15, aby krzywa cosinusa LR wyglądała ciekawiej
LEARNING_RATE = 0.003 # Zaczynamy od wyższego LR, bo scheduler będzie go zmniejszał
DATA_DIR = './data'

# 1. Definicja własnego szumu
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# 2. Augmentacja danych
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    print("Wczytywanie datasetu...")
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    classes = full_dataset.classes
    print(f"Znalezione klasy: {classes}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Przygotowanie Modelu
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)

    # 3. Lepszy Optymalizator (AdamW) i Scheduler LR
    criterion = nn.CrossEntropyLoss()
    # AdamW posiada lepszą regularyzację wagi (weight_decay), świetnie działa w wizji
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # CosineAnnealing płynnie zmniejsza Learning Rate po cosinusoidzie w trakcie epok
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Słownik do śledzenia historii dla wykresów
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    # Pętla Uczenia
    print("\nRozpoczynamy trening...")
    for epoch in range(EPOCHS):
        # Faza Treningowa
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Faza Walidacyjna
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Aktualny Learning Rate ZANIM zrobimy krok schedulera
        current_lr = optimizer.param_groups[0]['lr']
        
        # Zapisywanie metryk do wykresów
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Krok schedulera - modyfikacja LR na następną epokę
        scheduler.step()
        
        print(f"Epoka [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | "
              f"Strata T/W: {train_loss:.4f}/{val_loss:.4f} | "
              f"Dokładność T/W: {train_acc:.2f}% / {val_acc:.2f}%")

    print("\nTrening zakończony pomyślnie!")
    torch.save(model.state_dict(), "guitar_classifier.pth")
    print("Zapisano model jako 'guitar_classifier.pth'")
    
    # Wywołanie funkcji rysującej na samym końcu
    plot_training_history(history, EPOCHS)

if __name__ == '__main__':
    main()