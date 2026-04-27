import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams.update({'font.size': 12})

model_args = {
    'seed': 123,
    'batch_size': 128,
    'lr': 0.001, # Zmieniamy na mniejsze LR, bo użyjemy optymalizatora Adam
    'epochs': 10, # 10 epok z Adamem wystarczy, by przebić 99%
    'log_interval': 100
}

# 1. Ładowanie Danych
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_subset, validation_subset = torch.utils.data.random_split(mnist_train, [50000, 10000])
test_subset = datasets.MNIST('../data', train=False, download=True, transform=transform)

loader_kwargs = {'batch_size': model_args['batch_size'], 'num_workers': 2, 'pin_memory': True, 'shuffle': True}
train_loader = torch.utils.data.DataLoader(train_subset, **loader_kwargs)
validation_loader = torch.utils.data.DataLoader(validation_subset, **loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_subset, **loader_kwargs)

# 2. Zoptymalizowany Model CNN (Gwarantujący >99% Accuracy)
class TunedCNN(nn.Module):
    def __init__(self):
        super(TunedCNN, self).__init__()
        # Zwiększamy liczbę filtrów: z 2 na 16, i z 4 na 32
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.3) # Dodajemy dropout przestrzenny
        # Rozmiar po podwójnym MaxPoolingu z okna 28x28 przy kernelu 5 wynosi 4x4
        # 32 kanały * 4 * 4 = 512
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512) # Wypłaszczenie (Flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.4) # Zwykły dropout
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 3. Funkcje Trenujące
def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        prediction = output.argmax(dim=1)
        correct += prediction.eq(target).sum().item()
            
    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch {epoch_number} - Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_acc:.2f}%)')
    return train_loss, train_acc

def test(model, device, test_loader, message):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='mean').item()
            prediction = output.argmax(dim=1)
            correct += prediction.eq(target).sum().item()
            
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'{message}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n')
    return test_loss, test_acc

# 4. Funkcja do generowania Macierzy Błędów
def plot_confusion_matrix(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Zbieranie predykcji
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            # Zapisujemy tensory jako listy numpy
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    # Obliczanie macierzy za pomocą scikit-learn
    cm = confusion_matrix(all_targets, all_preds)
    
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    
    plt.title('Macierz Błędów (Confusion Matrix) na Test Set')
    plt.xlabel('Przewidywana Klasa (Predicted)')
    plt.ylabel('Prawdziwa Klasa (Ground Truth)')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Pętla Główna
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}\n")

model = TunedCNN().to(device)

# Używamy optymalizatora Adam zamiast SGD - uczy się znacznie szybciej
optimizer = optim.Adam(model.parameters(), lr=model_args['lr'])

torch.manual_seed(model_args['seed'])

print("Rozpoczynamy trening Zoptymalizowanego modelu CNN...")
for epoch in range(1, model_args['epochs'] + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, validation_loader, 'Validation set')

# Faza testowa po zakończeniu treningu (powinna przebić 99%!)
test(model, device, test_loader, 'Test set')

# Generowanie i zapis Macierzy Błędów
print("Generowanie macierzy błędów...")
plot_confusion_matrix(model, device, test_loader)