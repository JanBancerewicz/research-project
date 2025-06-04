import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

# Parametry
SEQ_LEN = 10  # długość sekwencji (okno czasowe)
BATCH_SIZE = 32
NUM_EPOCHS = 500
INPUT_SIZE = 5  # liczba cech
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASSES = 2

# Wczytanie i przygotowanie sekwencji (okien czasowych)
def create_sequences(features, labels, seq_len=SEQ_LEN):
    sequences = []
    seq_labels = []
    for i in range(len(features) - seq_len + 1):
        seq = features[i:i+seq_len]
        label = labels[i+seq_len-1]  # etykieta przypisana do ostatniego elementu w sekwencji
        sequences.append(seq)
        seq_labels.append(label)
    return np.array(sequences), np.array(seq_labels)

# Wczytanie i skalowanie danych
def load_and_prepare(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    label_map = {'exhale': 0, 'inhale': 1}
    train_df['label'] = train_df['breath_state'].map(label_map)
    test_df['label'] = test_df['breath_state'].map(label_map)

    features_train = train_df[['rmssd', 'sdnn', 'hr', 'edr_mean', 'rr_slope']].values
    features_test = test_df[['rmssd', 'sdnn', 'hr', 'edr_mean', 'rr_slope']].values

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    labels_train = train_df['label'].values
    labels_test = test_df['label'].values

    # Tworzymy sekwencje
    X_train, y_train = create_sequences(features_train, labels_train, SEQ_LEN)
    X_test, y_test = create_sequences(features_test, labels_test, SEQ_LEN)

    return X_train, y_train, X_test, y_test

# Dataset
class BreathSeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Model LSTM
class BreathLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super(BreathLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Normalizacja i dropout po LSTM
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.4)

        # Dodatkowa warstwa dense + aktywacja
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)

        # Warstwa wyjściowa
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # bierzemy ostatni krok sekwencji

        out = self.batchnorm1(out)
        out = self.dropout1(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout2(out)

        out = self.fc2(out)  # logits
        return out

# Ładowanie danych
train_file = 'data_finall.csv'
test_file = 'data.csv'

X_train, y_train, X_test, y_test = load_and_prepare(train_file, test_file)

train_dataset = BreathSeqDataset(X_train, y_train)
test_dataset = BreathSeqDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inicjalizacja modelu i optymalizatora
model = BreathLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for seqs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# Ewaluacja
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for seqs, labels in test_loader:
        outputs = model(seqs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy*100:.2f}%")
