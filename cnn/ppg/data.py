import os
import glob
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

from cnn.ppg.PPGPeakDetector import PPGPeakDetector

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
    df = pd.read_csv(filepath)
    ppg = df['ppg'].values[20:]
    fs = 30  # Sampling frequency (Hz)
    ppg_bp = butter_bandpass_filter(ppg, 0.5, 5.0, fs=fs, order=3)
    if len(ppg_bp) < segment_length:
        return np.empty((0, segment_length)), np.empty((0, segment_length))
    segments, labels = [], []
    for i in range(len(ppg_bp) - segment_length + 1):
        segment = ppg_bp[i:i+segment_length]
        # Normalize each segment (window) individually
        segment_norm = normalize_signal(segment)
        # Find peaks inside this segment
        distance_samples = int(round(0.35 * fs))
        prominence = np.std(segment) * 0.5
        peak_indices, _ = find_peaks(segment, distance=distance_samples, prominence=prominence)
        label = np.zeros(segment_length)
        label[peak_indices] = 1
        segments.append(segment_norm)
        labels.append(label)
    return np.array(segments), np.array(labels)

def load_ppg_segments_from_directory(directory, segment_length=50, max_files=None):
    all_segments = []
    all_labels = []
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if max_files:
        csv_files = csv_files[:max_files]
    for csv_path in csv_files:
        segments, labels = load_ppg_segments_from_csv(csv_path, segment_length=segment_length)
        # Only add if there are enough samples for at least one segment
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
        if max_segments:
            X = X[:max_segments]
            y = y[:max_segments]
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, segment_length)
        self.y = torch.tensor(y, dtype=torch.float32)               # (N, segment_length)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model Training/Evaluation ---

def train_model(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

def test_model(model, dataset, num_windows=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_ppg, all_true, all_pred = [], [], []
    correct = 0
    all_pred_peaks = 0
    all_true_peaks = 0
    missed = 0
    for i in range(min(num_windows, len(dataset))):
        X_sample, y_true = dataset[i]
        X_tensor = X_sample.unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(X_tensor).squeeze().cpu().numpy()
        all_ppg.extend(X_sample.squeeze().numpy())
        all_true.extend(y_true.numpy())
        pred_binary = (y_pred > 0.5).astype(int)
        all_pred.extend(pred_binary)
        # Stats calculation
        for j in range(len(pred_binary)):
            if pred_binary[j] == y_true[j].item() and y_true[j].item() == 1:
                correct += 1
            if y_true[j].item() == 1:
                all_true_peaks += 1
            if pred_binary[j] == 1:
                all_pred_peaks += 1
            if pred_binary[j] == 0 and y_true[j].item() == 1:
                missed += 1
    # Print stats like r_neural.py
    if all_true_peaks > 0:
        p = correct / all_true_peaks * 100
        p2 = max((all_pred_peaks / all_true_peaks * 100) - 100, 0)
        p3 = missed / all_true_peaks * 100
        print(f"Accuracy: {p:.2f}%")
        print(f"Additional: {p2:.2f}%")
        print(f"Missed: {p3:.2f}%")
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred)
    print("\nConfusion matrix:")
    print(cm)
    f1 = f1_score(all_true, all_pred)
    print(f"\nF1-score: {f1:.4f}")


def get_or_train_model(
    model_path,
    data_dir,
    segment_length=50,
    max_segments=10000,
    epochs=10,
    batch_size=32,
    lr=0.0001,
    max_files=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        print(f"📦 Loading model from {model_path}")
        model = PPGPeakDetector()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    print("🚀 Training new model...")
    dataset = PPGDirectoryDataset(data_dir, segment_length=segment_length, max_segments=max_segments, max_files=max_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = PPGPeakDetector()
    start = time.time()
    train_model(model, dataloader, epochs=200, lr=0.001)
    test_model(model, dataset)
    torch.save(model.state_dict(), model_path)
    end = time.time()
    print(f"Execution time: {end - start:.6f} seconds")
    print(f"💾 Model saved to {model_path}")
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

    output_array = output_array > 0.5  # Convert probabilities to binary predictions
    return  output_array
