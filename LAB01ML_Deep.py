import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 2. Generowanie danych treningowych
X_list = []
Y_list = []
X_kon = []
Y_kon = []
for a in np.arange(-1.5, 1.5, 0.0001):
    features = [a]
    
    # Wynikiem jest po prostu jedna liczba zmiennoprzecinkowa
    sum_ab = np.sin(70*a) * np.exp(a)
    target = [sum_ab]
        
    X_list.append(features)
    Y_list.append(target)

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32)

# 3. Definicja modelu (Teraz to sieć do regresji, nie klasyfikacji)
class ContinuousPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContinuousPredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.tanh(x)
        x = self.hidden2(x)
        x = self.tanh(x)
        x = self.output_layer(x)
        return x # Zwracamy czystą, liniową wartość

# 4. Inicjalizacja modelu i trening
INPUT_SIZE = 1
OUTPUT_SIZE = 1
hidden_size = 32
model = ContinuousPredictor(INPUT_SIZE, hidden_size, OUTPUT_SIZE)

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32)

# Sprawdzenie dostępności GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używam urządzenia: {device}")

# Przeniesienie modelu i danych na urządzenie
model = model.to(device)
X = X.to(device)
Y = Y.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
MSE = nn.MSELoss()

N_epochs = 3500

print("Rozpoczynam trening")
model.train()
for epoch in range(N_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = MSE(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
         print(f"Epoka {epoch}, Loss: {loss.item()}")

print(f"Trening zakończony. Końcowy Loss: {loss.item()}\n")

# 6. Generowanie przewidywań dla całego zakresu
model.eval()
with torch.no_grad():
    Y_pred = model(X).cpu().numpy()

# 7. Testowanie działania
test_values = [0.0, 1.5, -1.5, -1.0, 1.0] 

print("--- Wyniki Testów ---")
for a_val in test_values:
    x_test = torch.tensor([[a_val]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_val = model(x_test)[0].item() 
        
    prawidlowy_wynik = np.sin(70*a_val) * np.exp(a_val)
    
    print(f"Dane wejściowe a={a_val}")
    print(f"Odpowiedź sieci:  {pred_val:.4f}")
    print(f"Prawidłowy wynik: {prawidlowy_wynik:.4f}")
    print("-" * 20)
    X_kon.append(a_val)
    Y_kon.append(pred_val)
   


plt.figure(figsize=(10,6))
plt.plot(X_list, Y_list, label='Wartości liczone przez komputer', color='blue', linewidth=2)
plt.plot(X_list, Y_pred, label='Program do Nauczania Maszynowego', color='orange', linewidth=2)
plt.legend()
plt.show()