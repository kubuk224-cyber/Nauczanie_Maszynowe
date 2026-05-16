import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# Parametry uczenia
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = './data' # Ścieżka do Twojego głównego folderu z danymi

# 1. Definicja własnego szumu (Gaussian Noise)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Dodajemy szum do tensora i ucinamy wartości do zakresu [0, 1]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# 2. Augmentacja danych (Data Augmentation)
# Dla zestawu treningowego dodajemy obroty, szumy i odbicia lustrzane
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet wymaga zdjęć w rozmiarze 224x224
    transforms.RandomRotation(degrees=30), # Losowy obrót od -30 do 30 stopni
    transforms.RandomHorizontalFlip(p=0.5), # Losowe odbicie lustrzane
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Delikatne zmiany kolorów
    transforms.ToTensor(), # Zamiana obrazka na Tensor
    AddGaussianNoise(0., 0.05), # Nałożenie naszego szumu Gaussa
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacja ImageNet
])

# Dla zestawu walidacyjnego TYLKO zmieniamy rozmiar i normalizujemy (bez szumów!)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    # 3. Ładowanie danych
    print("Wczytywanie datasetu...")
    # Wczytujemy cały dataset (ImageFolder sam ogarnie podfoldery jako klasy)
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    
    # Wyświetlanie znalezionych klas
    classes = full_dataset.classes
    print(f"Znalezione klasy: {classes}")

    # Podział na zbiór treningowy (80%) i walidacyjny (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Nadpisanie transformacji dla zbioru walidacyjnego (nie chcemy tam szumów)
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Przygotowanie Modelu (Transfer Learning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Pobieramy wytrenowany model ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Zamrażamy wagi bazowe (nie chcemy zepsuć tego, czego ResNet już się nauczył)
    for param in model.parameters():
        param.requires_grad = False

    # Zamieniamy tylko ostatnią warstwę klasyfikacyjną
    # ResNet18 ma w oryginalnej ostatniej warstwie `fc.in_features` neuronów wejściowych
    num_ftrs = model.fc.in_features
    # Dajemy 2 neurony na wyjście, bo mamy 2 klasy (Single_cut, Double_cut)
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)

    # 5. Funkcja straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    # Uczymy TYLKO ostatnią warstwę (model.fc.parameters())
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 6. Pętla Uczenia
    print("Rozpoczynamy trening...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
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
            
        train_acc = 100 * correct_train / total_train
        
        # 7. Walidacja po każdej epoce
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoka [{epoch+1}/{EPOCHS}] - Strata: {running_loss/len(train_loader):.4f} | Dokładność Trening: {train_acc:.2f}% | Dokładność Walidacja: {val_acc:.2f}%")

    print("Trening zakończony pomyślnie!")
    
    # Zapisz model
    torch.save(model.state_dict(), "guitar_classifier.pth")
    print("Zapisano model jako 'guitar_classifier.pth'")

if __name__ == '__main__':
    main()