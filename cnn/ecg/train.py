import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys  # Dodany import sys dla pewności
from cnn.ecg.ECG_CNN import ECG_CNN, split_into_chunks, MODEL_PATH

# --- KONFIGURACJA PODZIAŁU DANYCH (DATA SPLIT) ---
# Pliki, które zostaną wykluczone z treningu (rezerwujemy je do testów)
# Zmieniamy to tutaj, aby zapobiec Data Leakage.
TEST_FILES = [
    "R20.csv",
    "R21.csv"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 256)  # Dopasowanie do wyjścia modelu

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")


def init_model():
    model = ECG_CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_signals = np.empty((0, 256), dtype=np.float32)
    train_labels = np.empty((0, 256), dtype=np.float32)

    # --- PATH FIX START ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir))
    data_dir = os.path.join(project_root, "data", "r")

    print(f"DEBUG: Looking for ECG training data in: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"CRITICAL: Directory not found: {data_dir}")

    paths = get_csv_file_paths(data_dir)

    if not paths:
        raise ValueError(f"CRITICAL: No CSV files found in {data_dir}. Check your data folder.")
    # --- PATH FIX END ---

    print(f"--- PREPARING DATASET (Excluding Test Files: {TEST_FILES}) ---")
    files_used = 0

    for p in paths:
        file_name = os.path.basename(p)

        # --- DATA LEAKAGE PROTECTION ---
        if file_name in TEST_FILES:
            print(f"[-] SKIPPING TEST FILE: {file_name} (Reserved for validation)")
            continue
        # -------------------------------

        try:
            print(f"[+] Loading training file: {file_name}")
            df = pd.read_csv(p)
            train_signal = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
            train_label = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)

            # Walidacja poprawności chunków (czasem ostatni jest krótszy)
            if train_signal.shape[1] == 256:
                train_signals = np.concatenate((train_signals, train_signal))
                train_labels = np.concatenate((train_labels, train_label))
                files_used += 1
            else:
                print(f"Warning: Skipping file {file_name} due to shape mismatch: {train_signal.shape}")

        except Exception as e:
            print(f"Error loading file {p}: {e}")

    if len(train_signals) == 0:
        raise ValueError("Train signals are empty! Check if you didn't exclude all files.")

    print(f"Dataset ready. Used {files_used} files. Total samples: {len(train_signals)}")

    train_dataset = TensorDataset(torch.tensor(train_signals).unsqueeze(1), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("🚀 Starting ECG model training...")
    train_model(model, train_loader, criterion, optimizer, epochs=50)

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