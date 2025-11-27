import os
import glob
import time
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from scipy.signal import butter, filtfilt, find_peaks

# Import modelu - zabezpieczenie importu
try:
    from cnn.ppg.PPGPeakDetector import PPGPeakDetector
except ImportError:
    # Fallback jeśli uruchamiamy bezpośrednio z folderu ppg
    from PPGPeakDetector import PPGPeakDetector

# --- KONFIGURACJA PODZIAŁU DANYCH (DATA SPLIT) ---
# Pliki przeznaczone WYŁĄCZNIE do walidacji (nie biorą udziału w treningu)
VAL_FILES = [
    "ppg_data_johnny_10min.csv"
]


# --- Utility Functions ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return 2 * (signal - min_val) / (max_val - min_val) - 1  # scale to [-1, 1]


def load_ppg_segments_from_csv(filepath, segment_length=50):
    try:
        df = pd.read_csv(filepath)
        # Obsługa różnych nazw kolumn
        if 'ppg' in df.columns:
            raw = df['ppg'].values
        elif 'signal' in df.columns:
            raw = df['signal'].values
        else:
            return np.empty((0, segment_length)), np.empty((0, segment_length))

        if len(raw) > 20:
            ppg = raw[20:]
        else:
            return np.empty((0, segment_length)), np.empty((0, segment_length))

        fs = 30
        ppg_bp = butter_bandpass_filter(ppg, 0.5, 5.0, fs=fs, order=3)

        if len(ppg_bp) < segment_length:
            return np.empty((0, segment_length)), np.empty((0, segment_length))

        segments, labels = [], []
        for i in range(len(ppg_bp) - segment_length + 1):
            segment = ppg_bp[i:i + segment_length]
            segment_norm = normalize_signal(segment)

            distance_samples = int(round(0.35 * fs))
            prominence = np.std(segment) * 0.5
            peak_indices, _ = find_peaks(segment, distance=distance_samples, prominence=prominence)

            label = np.zeros(segment_length)
            label[peak_indices] = 1
            segments.append(segment_norm)
            labels.append(label)

        return np.array(segments), np.array(labels)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.empty((0, segment_length)), np.empty((0, segment_length))


def load_ppg_segments_from_directory(directory, segment_length=50, max_files=None, mode='train'):
    """
    mode: 'train' (exclude VAL_FILES) or 'val' (only use VAL_FILES) or 'all'
    """
    all_segments = []
    all_labels = []

    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return np.empty((0, segment_length)), np.empty((0, segment_length))

    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    if not csv_files:
        print(f"WARNING: No CSV files in {directory}")

    files_loaded_count = 0
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)

        # --- FILTROWANIE PLIKÓW ---
        if mode == 'train':
            if filename in VAL_FILES:
                continue  # Skip validation file in training
        elif mode == 'val':
            if filename not in VAL_FILES:
                continue  # Skip non-validation files
        # --------------------------

        segments, labels = load_ppg_segments_from_csv(csv_path, segment_length=segment_length)
        if len(segments) > 0:
            all_segments.append(segments)
            all_labels.append(labels)
            files_loaded_count += 1
            print(f"[{mode.upper()}] Loaded: {filename} ({len(segments)} segments)")

        if max_files and files_loaded_count >= max_files:
            break

    if all_segments:
        X = np.vstack(all_segments)
        y = np.vstack(all_labels)
    else:
        X = np.empty((0, segment_length))
        y = np.empty((0, segment_length))
    return X, y


# --- Dataset ---

class PPGDirectoryDataset(Dataset):
    def __init__(self, directory, segment_length=50, max_segments=None, max_files=None, mode='train'):
        X, y = load_ppg_segments_from_directory(directory, segment_length=segment_length, max_files=max_files,
                                                mode=mode)

        if len(X) == 0:
            self.X = torch.empty(0)
            self.y = torch.empty(0)
        else:
            if max_segments:
                X = X[:max_segments]
                y = y[:max_segments]
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Model Training/Evaluation ---

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"\nStarting training on {device}...")

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)

        avg_train_loss = train_loss / train_count if train_count > 0 else 0

        # --- VALIDATION PHASE ---
        val_loss = 0.0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                out_val = model(X_val)
                loss = criterion(out_val, y_val)

                val_loss += loss.item() * X_val.size(0)
                val_count += X_val.size(0)

        avg_val_loss = val_loss / val_count if val_count > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


def get_or_train_model(
        model_path,
        data_dir,
        segment_length=100,
        max_segments=10000,
        epochs=10,
        batch_size=32,
        lr=0.0001,
        max_files=None,
        num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PATH FIX ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(current_script_dir))
        data_dir_abs = os.path.join(current_script_dir, "train_data")
        model_path_abs = os.path.join(project_root, model_path)
    else:
        model_path_abs = model_path
        data_dir_abs = data_dir

    if not os.path.exists(data_dir) and os.path.exists(os.path.join(current_script_dir, "train_data")):
        data_dir = os.path.join(current_script_dir, "train_data")
    # ----------------

    # Sprawdzenie czy model istnieje
    final_model_path = model_path if os.path.exists(model_path) else (
        model_path_abs if os.path.exists(model_path_abs) else None)

    if final_model_path:
        print(f"📦 Loading model from {final_model_path}")
        model = PPGPeakDetector()
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")

    print("🚀 Training new model with Validation Split...")
    print(f"Data directory: {data_dir}")
    print(f"Validation file: {VAL_FILES}")

    # 1. Train Dataset (bez pliku Johnny'ego)
    train_dataset = PPGDirectoryDataset(
        data_dir,
        segment_length=segment_length,
        max_segments=max_segments,
        max_files=max_files,
        mode='train'
    )

    # 2. Validation Dataset (tylko plik Johnny'ego)
    val_dataset = PPGDirectoryDataset(
        data_dir,
        segment_length=segment_length,
        max_segments=max_segments,  # Można dać mniej do walidacji
        max_files=max_files,
        mode='val'
    )

    if len(train_dataset) == 0:
        raise ValueError("CRITICAL: Train dataset is empty!")
    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty! Check filenames.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))

    model = PPGPeakDetector()
    start = time.time()

    # Uruchamiamy trening z walidacją
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    save_path = model_path_abs if not os.path.isabs(model_path) else model_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)
    end = time.time()
    print(f"Execution time: {end - start:.6f} seconds")
    print(f"💾 Model saved to {save_path}")
    return model


def predict_ppg_segment(model, input_array):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    if input_array.ndim == 1:
        input_array = input_array[np.newaxis, np.newaxis, :]
    elif input_array.ndim == 2:
        input_array = input_array[np.newaxis, :]
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_array = output_tensor.squeeze().cpu().numpy()

    output_array = output_array > 0.5
    return output_array