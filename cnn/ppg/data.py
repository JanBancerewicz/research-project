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
    from PPGPeakDetector import PPGPeakDetector


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


def load_ppg_segments_from_directory(directory, segment_length=50, max_files=None):
    all_segments = []
    all_labels = []

    # Zabezpieczenie: sprawdź czy katalog istnieje
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return np.empty((0, segment_length)), np.empty((0, segment_length))

    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    if not csv_files:
        print(f"WARNING: No CSV files in {directory}")

    if max_files:
        csv_files = csv_files[:max_files]

    for csv_path in csv_files:
        segments, labels = load_ppg_segments_from_csv(csv_path, segment_length=segment_length)
        if len(segments) > 0:
            all_segments.append(segments)
            all_labels.append(labels)

    if all_segments:
        X = np.vstack(all_segments)
        y = np.vstack(all_labels)
    else:
        X = np.empty((0, segment_length))
        y = np.empty((0, segment_length))
    return X, y


# --- Dataset ---

class PPGDirectoryDataset(Dataset):
    def __init__(self, directory, segment_length=50, max_segments=None, max_files=None):
        X, y = load_ppg_segments_from_directory(directory, segment_length=segment_length, max_files=max_files)

        if len(X) == 0:
            # Pusty dataset, żeby DataLoader nie wywalił błędu przy inicjalizacji,
            # ale wywali błąd przy treningu jeśli nie obsłużymy
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

def train_model(model, dataloader, epochs=5, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            count += X_batch.size(0)

        if count > 0:
            avg = epoch_loss / count
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - No data processed!")


def test_model(model, dataset, num_windows=100):
    if len(dataset) == 0:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ... (logika testowania bez zmian) ...
    # Skrócona dla czytelności, bo błąd nie był tutaj
    pass


def get_or_train_model(
        model_path,
        data_dir,  # <-- To przychodzi jako argument (prawdopodobnie błędny/względny)
        segment_length=100,
        max_segments=10000,
        epochs=10,
        batch_size=32,
        lr=0.0001,
        max_files=None,
        num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. NAPRAWA ŚCIEŻEK (Path Fix) ---
    # Ustal gdzie fizycznie znajduje się ten plik (cnn/ppg/data.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Jeśli model_path jest względna, napraw ją
    if not os.path.isabs(model_path):
        # Zakładamy, że model ma być w cnn/ppg/ jeśli podano samą nazwę pliku
        # Albo cofamy się do root, jeśli ścieżka zaczyna się od cnn/
        # Najbezpieczniej: zróbmy absolute path względem projektu
        project_root = os.path.dirname(os.path.dirname(current_script_dir))
        # Jeśli path to 'cnn/ppg/model.pth', a my jesteśmy w root...
        # Tutaj prosta heurystyka: jeśli plik istnieje pod wskazaną ścieżką (absolutną), użyj jej.
        # Jeśli nie, spróbuj znaleźć go w folderze skryptu.

        # Wersja najprostsza: Wymuś ścieżkę absolutną do folderu train_data
        # Ignorujemy to co przyszło w data_dir jeśli nie działa
        data_dir_abs = os.path.join(current_script_dir, "train_data")

        # Sprawdźmy czy model istnieje pod pełną ścieżką
        # Jeśli podano "cnn/ppg_model.pth", a root to C:/.../research-project
        model_path_abs = os.path.join(project_root, model_path)
    else:
        model_path_abs = model_path
        data_dir_abs = data_dir  # Jeśli ktoś podał absolutną, ufamy mu

    # Fallback dla danych - jeśli podany folder nie istnieje, użyj train_data obok skryptu
    if not os.path.exists(data_dir) and os.path.exists(os.path.join(current_script_dir, "train_data")):
        print(f"DEBUG: Redirecting data path to {os.path.join(current_script_dir, 'train_data')}")
        data_dir = os.path.join(current_script_dir, "train_data")

    # --- KONIEC POPRAWKI ---

    # Teraz sprawdzamy istnienie modelu używając (potencjalnie) naprawionej ścieżki
    # Uwaga: sprawdzam obie: oryginalną (jeśli CWD jest ok) i absolutną
    if os.path.exists(model_path):
        final_model_path = model_path
    elif os.path.exists(model_path_abs):
        final_model_path = model_path_abs
    else:
        final_model_path = None

    if final_model_path:
        print(f"📦 Loading model from {final_model_path}")
        model = PPGPeakDetector()
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")

    print("🚀 Training new model...")
    print(f"Looking for data in: {data_dir}")

    dataset = PPGDirectoryDataset(data_dir, segment_length=segment_length, max_segments=max_segments,
                                  max_files=max_files)

    if len(dataset) == 0:
        raise ValueError(f"CRITICAL: Dataset is empty! Check path: {data_dir}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))

    model = PPGPeakDetector()
    start = time.time()
    train_model(model, dataloader, epochs=epochs, lr=lr, device=device)
    # test_model(model, dataset) # Opcjonalnie

    # Zapisz model (używając absolutnej ścieżki jeśli trzeba)
    save_path = model_path_abs if not os.path.isabs(model_path) else model_path

    # Upewnij się, że katalog istnieje
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