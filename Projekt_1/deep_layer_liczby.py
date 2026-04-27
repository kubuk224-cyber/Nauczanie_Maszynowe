import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

model_args = {}
# random seed
model_args['seed'] = 123
# batch size of 128 in Stochastic Gradient Descent (SGD)
model_args['batch_size'] = 128
# learning rate
model_args['lr'] = .05
# SGD momentum
model_args['momentum'] = .5
# number of epochs
model_args['epochs'] = 50
# logging frequency
model_args['log_interval'] = 100

# load the MNIST dataset via torchvision
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)

# divide data into training and validation subsets
train_subset, validation_subset = torch.utils.data.random_split(mnist_train, [50000, 10000])
test_subset = datasets.MNIST('../data', train=False, download=True, transform=transform)

# define dataloaders
loader_kwargs = {'batch_size': model_args['batch_size'],
                 'num_workers': 2,
                 'pin_memory': True,
                 'shuffle': True}
train_loader = torch.utils.data.DataLoader(train_subset, **loader_kwargs)
validation_loader = torch.utils.data.DataLoader(validation_subset, **loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_subset, **loader_kwargs)

print("Train subset size:", len(train_subset))
print("Validation subset size:", len(validation_subset))
print("Test subset size:", len(test_subset))
print("Train batches:", len(train_loader)) 

example_number = 123
print("Single item shape:", train_subset[example_number][0][0].shape)

# Wstępna wizualizacja kilku próbek z MNIST
fig, axs = plt.subplots(5, 5, figsize=(7,7), tight_layout=True)
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(train_subset[example_number+i*5+j][0].reshape(28,28), cmap='gray')
        axs[i,j].set_title(train_subset[example_number+i*5+j][1])
        axs[i,j].axis('off') # wyłączenie osi dla lepszego wyglądu
plt.show() # Wyświetlamy siatkę obrazków przed startem treningu


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class Deep(nn.Module):
    def __init__(self):
        super(Deep, self).__init__()
        self.fc1 = nn.Linear(28*28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    correct = 0 # Zmienna do zliczania poprawnych odpowiedzi
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Obliczamy poprawność dla bieżącego batcha
        prediction = output.argmax(dim=1)
        correct += prediction.eq(target).sum().item()
        
        if batch_idx % model_args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset), train_accuracy))
    
    return train_loss, train_accuracy

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
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset), test_accuracy))
    
    return test_loss, test_accuracy

def plot_metrics(train_loss, validation_loss, train_acc, validation_acc, title):
    # Inicjalizacja płótna z dwoma wykresami obok siebie
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Wykres 1: Strata (Loss)
    ax1.plot(range(1, len(train_loss)+1), train_loss, 'o-', label='Trening', color='blue')
    ax1.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', label='Walidacja', color='orange')
    ax1.set_title('Wykres błędu (Loss)')
    ax1.set_xlabel('Epoki')
    ax1.set_ylabel('Średnia strata')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Wykres 2: Dokładność (Accuracy)
    ax2.plot(range(1, len(train_acc)+1), train_acc, 'o-', label='Trening', color='green')
    ax2.plot(range(1, len(validation_acc)+1), validation_acc, 'o-', label='Walidacja', color='red')
    ax2.set_title('Wykres dokładności (Accuracy)')
    ax2.set_xlabel('Epoki')
    ax2.set_ylabel('Dokładność [%]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Zapis i wyświetlenie
    fig.suptitle(f'Postępy uczenia: {title}', fontsize=16, fontweight='bold')
    plt.savefig('cnn_historia_zaawansowana.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Ustawienie urządzenia sprzętowego (GPU jeśli dostępne, inaczej CPU)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}\n")

# Inicjalizacja modelu CNN na wybrane urządzenie
model = CNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=model_args['lr'], momentum=model_args['momentum'])

torch.manual_seed(model_args['seed'])

# Listy do zbierania historii z każdej epoki
train_loss_history = []
validation_loss_history = []
train_acc_history = []
validation_acc_history = []

print("Rozpoczynamy trening...")
for epoch_number in range(1, model_args['epochs'] + 1):
    t_loss, t_acc = train(model, device, train_loader, optimizer, epoch_number)
    v_loss, v_acc = test(model, device, validation_loader, 'Validation set')
    
    train_loss_history.append(t_loss)
    train_acc_history.append(t_acc)
    
    validation_loss_history.append(v_loss)
    validation_acc_history.append(v_acc)

# Faza testowa po zakończeniu treningu
test(model, device, test_loader, 'Test set')

# Rysowanie ostatecznego wykresu i zapis do pliku
plot_metrics(train_loss_history, validation_loss_history, train_acc_history, validation_acc_history, 'Model CNN')