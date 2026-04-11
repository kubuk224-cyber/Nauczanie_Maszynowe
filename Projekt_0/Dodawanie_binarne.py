import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definicja problemu: Ustalmy maksymalną wielkość liczb na 4 bity
NUM_BITS = 4
INPUT_SIZE = NUM_BITS * 2   # 8 bitów na wejściu (dwie liczby 4-bitowe)
OUTPUT_SIZE = NUM_BITS + 1  # 5 bitów na wyjściu (maksymalny wynik to 30, czyli 11110)

# 2. Generowanie danych treningowych (wszystkie 256 kombinacji)
X_list = []
Y_list = []

for a in range(2**NUM_BITS):
    for b in range(2**NUM_BITS):
        # Formatowanie do postaci binarnej np. 3 -> '0011'
        bin_a = format(a, f'0{NUM_BITS}b')
        bin_b = format(b, f'0{NUM_BITS}b')
        
        # Wejście: połączone bity liczby A i B jako lista intów [0,0,1,1, 0,1,0,1]
        features = [int(bit) for bit in bin_a] + [int(bit) for bit in bin_b]
        
        # Wyjście: prawidłowy wynik dodawania w postaci binarnej
        sum_ab = a + b
        bin_sum = format(sum_ab, f'0{OUTPUT_SIZE}b')
        target = [int(bit) for bit in bin_sum]
        
        X_list.append(features)
        Y_list.append(target)

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32)

# 3. Definicja modelu (Deep Perceptron)
class BinaryAdder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryAdder, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # Zmiana na ReLU w warstwie ukrytej pomaga przy większych sieciach
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return self.sigmoid(x) # Zwraca wartości od 0 do 1 dla każdego z 5 bitów wyniku

# 4. Inicjalizacja modelu i trening
hidden_size = 32 # Zwiększamy warstwę ukrytą, bo mamy trudniejszy problem
model = BinaryAdder(INPUT_SIZE, hidden_size, OUTPUT_SIZE)

# Używamy optymalizatora Adam - uczy się szybciej i stabilniej niż SGD dla tego typu sieci
optimizer = optim.Adam(model.parameters(), lr=0.01)
MSE = nn.MSELoss()

N_epochs = 3000
train_loss = []

print("Rozpoczynam trening (może zająć kilka-kilkanaście sekund)...")
model.train()
for epoch in range(N_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = MSE(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
         print(f"Epoka {epoch}, Loss: {loss.item():.4f}")

print(f"Trening zakończony. Końcowy Loss: {loss.item():.6f}\n")

# 5. Testowanie działania (sprawdźmy kilka wybranych przykładów)
model.eval()
test_pairs = [(3, 2), (15, 15), (7, 8), (0, 0)] # Wybrane liczby dziesiętne do testu

print("--- Wyniki Testów ---")
for a_val, b_val in test_pairs:
    # Przygotowanie danych do formatu wejściowego sieci
    bin_a = format(a_val, f'0{NUM_BITS}b')
    bin_b = format(b_val, f'0{NUM_BITS}b')
    x_test = torch.tensor([[int(bit) for bit in bin_a] + [int(bit) for bit in bin_b]], dtype=torch.float32)
    
    # Predykcja
    with torch.no_grad():
        raw_pred = model(x_test)[0]
    
    # Progowanie (Thresholding): wszystko powyżej 0.5 staje się 1, poniżej 0
    binary_pred = [1 if val >= 0.5 else 0 for val in raw_pred]
    
    # Zamiana wyniku binarnego z powrotem na dziesiętny do wyświetlenia
    pred_str = "".join(str(bit) for bit in binary_pred)
    dec_pred = int(pred_str, 2)
    
    print(f"Dodajemy: {a_val} ({bin_a}) + {b_val} ({bin_b})")
    print(f"Odpowiedź sieci (binarnie): {pred_str}")
    print(f"Wynik dziesiętny: {dec_pred} (Poprawny: {a_val + b_val})\n")