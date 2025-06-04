import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os
from cnn.ecg.ECG_CNN import ECG_CNN, split_into_chunks, MODEL_PATH

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
    paths = get_csv_file_paths("./data/r")
    for p in paths:
        df = pd.read_csv(p)
        train_signal = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
        train_label = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)
        train_signals = np.concatenate((train_signals, train_signal))
        train_labels = np.concatenate((train_labels, train_label))

    train_dataset = TensorDataset(torch.tensor(train_signals).unsqueeze(1), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_model(model, train_loader, criterion, optimizer, epochs=20)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def get_csv_file_paths(directory):
    csv_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_file_paths.append(os.path.join(root, file))
    return csv_file_paths