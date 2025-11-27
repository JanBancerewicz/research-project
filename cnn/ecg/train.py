import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os
from cnn.ecg.ECG_CNN import ECG_CNN, split_into_chunks, MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, epochs=50):
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

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


def init_model():
    model = ECG_CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_signals = np.empty((0, 256), dtype=np.float32)
    train_labels = np.empty((0, 256), dtype=np.float32)

    # --- PATH FIX START ---
    # Ustalanie ścieżki absolutnej do folderu 'data/r'
    # 1. Gdzie jest ten plik (cnn/ecg/train.py)?
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Wyjście dwa poziomy wyżej do głównego katalogu projektu (research-project)
    project_root = os.path.dirname(os.path.dirname(current_script_dir))
    # 3. Pełna ścieżka do danych
    data_dir = os.path.join(project_root, "data", "r")

    print(f"DEBUG: Looking for ECG training data in: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"CRITICAL: Directory not found: {data_dir}")

    paths = get_csv_file_paths(data_dir)

    if not paths:
        raise ValueError(f"CRITICAL: No CSV files found in {data_dir}. Check your data folder.")
    # --- PATH FIX END ---

    for p in paths:
        try:
            df = pd.read_csv(p)
            train_signal = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
            train_label = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)
            train_signals = np.concatenate((train_signals, train_signal))
            train_labels = np.concatenate((train_labels, train_label))
        except Exception as e:
            print(f"Error loading file {p}: {e}")

    if len(train_signals) == 0:
        raise ValueError("Train signals are empty after loading! Check CSV content.")

    train_dataset = TensorDataset(torch.tensor(train_signals).unsqueeze(1), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("🚀 Starting ECG model training...")
    train_model(model, train_loader, criterion, optimizer, epochs=50)

    # Upewnij się, że folder docelowy dla modelu istnieje
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    return model


def get_csv_file_paths(directory):
    csv_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_file_paths.append(os.path.join(root, file))
    return csv_file_paths