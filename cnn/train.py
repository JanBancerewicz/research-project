import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from cnn.ECG_CNN import ECG_CNN, split_into_chunks, MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

def init_model():
    model = ECG_CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_signals = np.empty((0, 256), dtype=np.float32)
    train_labels = np.empty((0, 256), dtype=np.float32)
    paths = [
        "data/night_R0.csv",
        "data/night_R1.csv",
        "data/night_R2.csv",
        "data/night_R3.csv",
        "data/night_R4.csv",
        "data/night_R5.csv",
        "data/night_R7.csv",
        "data/night_R8.csv",
        "data/night_R9.csv",
        "data/R4.csv",
        "data/R5.csv",
        "data/g1_R0.csv",
        "data/g2_R0.csv",
        "data/g3_R0.csv",
    ]
    for p in paths:
        df = pd.read_csv(p)  # Replace with your file path
        train_signal = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
        train_label = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)
        train_signals = np.concatenate((train_signals, train_signal))
        train_labels = np.concatenate((train_labels, train_label))

    train_dataset = TensorDataset(torch.tensor(train_signals).unsqueeze(1), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_model(model, train_loader, criterion, optimizer, epochs=100)
    torch.save(model.state_dict(), MODEL_PATH)
    return model