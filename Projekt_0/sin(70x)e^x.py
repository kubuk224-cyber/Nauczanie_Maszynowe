import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. GENEROWANIE DANYCH
# Tworzymy 1000 punktów w przedziale [-1.5, 1.5]
X_cpu = torch.linspace(-1.5, 1.5, 1000).unsqueeze(1)
# Nasza docelowa funkcja
y_cpu = torch.sin(70 * X_cpu) * torch.exp(X_cpu)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
X, y = X_cpu.to(device), y_cpu.to(device)

# 2. DODATEK: CECHY FOURIERA
class FourierFeatures(nn.Module):
    def __init__(self, frequencies):
        super().__init__()
        # Zapisujemy częstotliwości
        self.freqs = torch.tensor(frequencies, dtype=torch.float32).to(device)
        
    def forward(self, x):
        features = []
        for freq in self.freqs:
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        # Łączymy wszystkie sinusy i cosinusy w jeden długi wektor cech
        return torch.cat(features, dim=-1)

frequencies = [1.0, 10.0, 35.0, 70.0]
fourier_layer = FourierFeatures(frequencies)
input_features_count = len(frequencies) * 2  # Każda częstotliwość daje sin() i cos()

# 3. MODEL
class PerceptronDeeper(torch.nn.Module):
    # constructor - dodano zmienną input_size
    def __init__(self, input_size, hidden_size):
        super(PerceptronDeeper, self).__init__()
        self.hidden_layer_1st = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2nd = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()  
        
    # method called once the model is executed
    def forward(self, x):
        x = self.hidden_layer_1st(x)
        x = self.activation(x)
        x = self.hidden_layer_2nd(x)
        x = self.activation(x)
        return self.output_layer(x)

# Ustawiamy znacznie szerszą sieć (64 zamiast 4)
model = PerceptronDeeper(input_size=input_features_count, hidden_size=64).to(device)

# Zmiana z SGD na Adam (lepiej radzi sobie z takimi falami)
optimizer = optim.Adam(model.parameters(), lr=0.01)
MSE = nn.MSELoss()

# 4. TRENING
N_epochs = 10000 # Adam konwerguje znacznie szybciej, 10k powinno wystarczyć
train_loss = []

print(f"Trenowanie na urządzeniu: {device}")
model.train()  # turn on the training mode
for epoch in range(N_epochs):
    optimizer.zero_grad()
    
    # Najpierw przepuszczamy wejście przez "magiczne" cechy Fouriera
    X_encoded = fourier_layer(X)
    
    # make predictions
    y_pred = model(X_encoded)
    
    # calculate the MSE loss
    loss = MSE(y_pred, y)
    
    # backpropagate the loss
    loss.backward()
    
    # update the model weights
    optimizer.step()
    train_loss.append(loss.item())
    
    # Co 1000 epok wypisujemy postęp
    if epoch % 1000 == 0:
        print(f"Epoka {epoch}, Loss: {loss.item():.4f}")

# 5. WYKRESY
# Wykres Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.title('Błąd w czasie (Loss)')
plt.xlabel('epoch')
plt.yscale('log')

# Wykres Predykcji
model.eval()
with torch.no_grad():
    X_encoded_test = fourier_layer(X)
    y_pred_final = model(X_encoded_test)

plt.subplot(1, 2, 2)
plt.plot(X.cpu(), y_pred_final.cpu(), label='Predykcja (Sieć)', color='red')
plt.plot(X.cpu(), y.cpu(), 'g', label='Wartość Prawdziwa', alpha=0.6)
plt.title('Dopasowanie do f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('wykres.png') # Zapisuje do pliku
print("Wykres został zapisany do pliku 'wykres.png' w folderze z kodem!")

# 6. WERYFIKACJA DLA WYBRANYCH PUNKTÓW
print("\n--- Wyniki predykcji dla wybranych punktów ---")

# Definiujemy punkty testowe
test_points = [-1.5, 0.0, 1.5]

# Zamieniamy na tensor o kształcie [3, 1] i wysyłamy na GPU/CPU
X_test_specific = torch.tensor(test_points, dtype=torch.float32).unsqueeze(1).to(device)

model.eval() # Upewniamy się, że model jest w trybie ewaluacji
with torch.no_grad():
    # 1. Transformacja Fouriera dla naszych 3 punktów
    X_test_encoded = fourier_layer(X_test_specific)
    
    # 2. Predykcja sieci
    y_pred_specific = model(X_test_encoded)
    
    # 3. Prawdziwe wartości matematyczne do porównania
    y_true_specific = torch.sin(70 * X_test_specific) * torch.exp(X_test_specific)
    
    # 4. Wyświetlanie wyników w pętli
    for i in range(len(test_points)):
        x_val = test_points[i]
        siec_val = y_pred_specific[i].item()
        prawda_val = y_true_specific[i].item()
        blad = abs(siec_val - prawda_val)
        
        print(f"x = {x_val:>4.1f} | Predykcja: {siec_val:>8.4f} | Prawda: {prawda_val:>8.4f} | Błąd: {blad:.5f}")
print("-" * 46)