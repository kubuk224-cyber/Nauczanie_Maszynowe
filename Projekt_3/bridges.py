import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import os
from PIL import Image

# ==========================================
# 1. PARAMETRY KONFIGURACYJNE
# ==========================================
BATCH_SIZE = 16  # Mniejszy batch, bo używamy większego modelu (ResNet-50)
EPOCHS = 15
LEARNING_RATE = 0.001
DATA_DIR = './data'

# Skrypt wejdzie TYLKO do tych folderów. double_cut i single_cut zostaną zignorowane.
ALLOWED_CLASSES = ['Floyd_sytle', 'Hardtail_style', 'Tremolo_style', 'Tune_o_matic_style']

# ==========================================
# 2. WŁASNY SZUM GAUSSA (AUGMENTACJA)
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Dodajemy szum i ucinamy wartości do poprawnego zakresu [0, 1]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# ==========================================
# 3. WŁASNA KLASA DATASETU (FILTROWANIE FOLDERÓW)
# ==========================================
class GuitarBridgesDataset(Dataset):
    def __init__(self, root_dir, allowed_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = allowed_classes
        # Przypisanie numerów klas (0, 1, 2, 3) do nazw folderów
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(allowed_classes)}
        
        self.image_paths = []
        self.labels = []

        # Przeszukujemy tylko dozwolone foldery
        for cls_name in allowed_classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Ostrzeżenie: Nie znaleziono folderu: {cls_dir}")
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 4. TRANSFORMACJE (AUGMENTACJA DANYCH)
# ==========================================
# Trening: Obracamy, szumimy, zmieniamy kolory, żeby sieć nie uczyła się na pamięć
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Walidacja: Tylko czyste zdjęcia, bez zniekształceń
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 5. FUNKCJA RYSUJĄCA WYKRESY
# ==========================================
def plot_metrics(history, epochs):
    print("\nGenerowanie wykresów treningu...")
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres Strata (Loss)
    ax1.plot(epochs_range, history['train_loss'], label='Trening (Train)', marker='o', color='blue')
    ax1.plot(epochs_range, history['val_loss'], label='Walidacja (Val)', marker='o', color='orange')
    ax1.set_title('Wykres Błędu (Loss)')
    ax1.set_xlabel('Epoki')
    ax1.set_ylabel('Strata')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Wykres Dokładność (Accuracy)
    ax2.plot(epochs_range, history['train_acc'], label='Trening (Train)', marker='o', color='green')
    ax2.plot(epochs_range, history['val_acc'], label='Walidacja (Val)', marker='o', color='red')
    ax2.set_title('Wykres Dokładności (Accuracy)')
    ax2.set_xlabel('Epoki')
    ax2.set_ylabel('Dokładność [%]')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('wykres_mostki.png', dpi=300)
    print("Zapisano wykres do pliku 'wykres_mostki.png'")
    plt.close()

# ==========================================
# 6. PĘTLA GŁÓWNA
# ==========================================
def main():
    print("Inicjalizacja Datasetu...")
    full_dataset = GuitarBridgesDataset(root_dir=DATA_DIR, allowed_classes=ALLOWED_CLASSES, transform=train_transforms)
    
    if len(full_dataset) == 0:
        print("Błąd: Dataset jest pusty! Sprawdź ścieżki i nazwy folderów.")
        return

    print(f"Zmapowane klasy: {full_dataset.class_to_idx}")
    print(f"Wczytano łącznie {len(full_dataset)} zdjęć mostków.")

    # Podział: 80% trening, 20% walidacja
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Klonujemy dataset dla walidacji, aby wyłączyć w niej augmentację
    val_dataset.dataset = GuitarBridgesDataset(root_dir=DATA_DIR, allowed_classes=ALLOWED_CLASSES, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie sprzętowe: {device}")

    # Transfer Learning: Używamy ResNet-50 (świetny do wyłapywania małych detali obrazu)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Zamrażamy stare wagi
    for param in model.parameters():
        param.requires_grad = False
    
    # Podmieniamy ostatnią warstwę (z 1000 na 4 klasy mostków)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(ALLOWED_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Nowoczesny optymalizator AdamW + weight decay (regularyzacja)
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    # Cosine Annealing łagodnie obniża Learning Rate pod koniec treningu
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Słownik do zapisywania wyników dla wykresu
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\nRozpoczynamy naukę modelu...")
    for epoch in range(EPOCHS):
        # --- TRENING ---
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
        
        # --- WALIDACJA ---
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
        
        # Zapisujemy dane do historii (do późniejszego wykresu)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Krok Schedulera (zmniejszamy Learning Rate na kolejną epokę)
        scheduler.step()
        
        print(f"Epoka [{epoch+1:02d}/{EPOCHS}] "
              f"Strata(T/W): {train_loss:.4f}/{val_loss:.4f} | "
              f"Skuteczność(T/W): {train_acc:.2f}% / {val_acc:.2f}%")

    print("\nTrening zakończony!")
    
    # Zapis wag modelu do pliku
    torch.save(model.state_dict(), "bridge_classifier.pth")
    print("Zapisano model jako 'bridge_classifier.pth'")
    
    # Wygenerowanie wykresu na koniec
    plot_metrics(history, EPOCHS)

if __name__ == '__main__':
    main()