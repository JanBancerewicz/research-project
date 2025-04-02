import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definicja modelu
class ECG_CNN(nn.Module):
    def __init__(self, window_size=256):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear((window_size // 8) * 64, 128)
        self.fc2 = nn.Linear(128, window_size)  # Wyjście 256 wartości

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # BCEWithLogitsLoss -> nie dodajemy Sigmoid
        return x

# Funkcja treningowa
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 256)  # Dopasowanie do wyjścia modelu

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Funkcja predykcji
def predict(model, sample):
    model.eval()
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample).squeeze(0)
        prediction = torch.sigmoid(output)
        binary_output = (prediction > 0.5).float().cpu().numpy()
    return binary_output

# Inicjalizacja modelu
model = ECG_CNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Przykładowe dane treningowe

df = pd.read_csv("data/R2.csv")  # Replace with your file path

def split_into_chunks(arr, chunk_size=256):
    arr = np.array(arr, dtype=float)  # Konwersja na numpy array
    num_full_chunks = len(arr) // chunk_size  # Liczba pełnych chunków
    return np.split(arr[:num_full_chunks * chunk_size], num_full_chunks)  # Dzielimy tylko pełne chunk

train_signals = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
train_labels = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)

train_dataset = TensorDataset(torch.tensor(train_signals).unsqueeze(1), torch.tensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Trening
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Testowanie predykcji

df = pd.read_csv("data/R1.csv")  # Replace with your file path

test_signals = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
test_labels = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)
# Funkcja testowa
def test_model(model, test_loader, criterion):
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 256)  # Dopasowanie do wyjścia modelu

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Obliczenie poprawnych predykcji (np. przy założeniu, że chcemy uzyskać dokładność)
            prediction = torch.sigmoid(outputs) > 0.5
            correct_predictions += (prediction == labels).sum().item()
            total_predictions += labels.numel()

    avg_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

# Przygotowanie testowego DataLoader
test_dataset = TensorDataset(torch.tensor(test_signals).unsqueeze(1), torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testowanie modelu
test_model(model, test_loader, criterion)

for i in range(5):
    sig = test_signals[i]
    prediction = predict(model, sig)
    print(f"Predykcja dla przykładowego sygnału:\n{prediction}")

    r = []
    t = []
    for i in range(len(prediction)):
        if prediction[i] == 1:
            t.append(i)
            r.append(sig[i])

    plt.plot(sig)
    plt.plot(t, r, "ro", label="Detected R-peaks")
    plt.show()