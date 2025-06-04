import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import numpy as np
from glob import glob
def prepare_rr_rmssd_from_csv(file_path, fs=130, window_sec=10, step_sec=2):
    df = pd.read_csv(file_path)
    ecg_signal = df['ecg'].values
    r_flags = df['R'].values
    r_peaks_idx = np.where(r_flags == 1)[0]
    rr_intervals = np.diff(r_peaks_idx) / fs
    r_times = r_peaks_idx / fs

    window_start = r_times[0]
    window_end = r_times[-1]
    rr_windows = []
    rmssd_values = []

    while window_start + window_sec <= window_end:
        mask = (r_times[:-1] >= window_start) & (r_times[:-1] < window_start + window_sec)
        rr_win = rr_intervals[mask]
        if len(rr_win) < 3:
            window_start += step_sec
            continue
        diff_rr = np.diff(rr_win)
        rmssd = np.sqrt(np.mean(diff_rr ** 2)) * 1000
        rr_windows.append(rr_win)
        rmssd_values.append(rmssd)
        window_start += step_sec

    max_len = max(len(w) for w in rr_windows) if rr_windows else 0
    return rr_windows, rmssd_values, max_len

def prepare_dataset_from_multiple_csvs( file_list, fs=130, window_sec=10, step_sec=2):
    rr_windows_all = []
    rmssd_all = []
    max_len_global = 0

    # Znajdź wszystkie pliki CSV w folderze


    # 1. Wczytaj i przygotuj dane z każdego pliku
    for file_path in file_list:
        rr_windows, rmssd, max_len = prepare_rr_rmssd_from_csv(file_path, fs, window_sec, step_sec)
        rr_windows_all.extend(rr_windows)
        rmssd_all.extend(rmssd)
        if max_len > max_len_global:
            max_len_global = max_len

    # 2. Padding do globalnego max_len
    rr_windows_padded = []
    for w in rr_windows_all:
        padded = np.pad(w, (0, max_len_global - len(w)), 'constant')
        rr_windows_padded.append(padded)

    rr_windows_array = np.array(rr_windows_padded, dtype=np.float32)
    rmssd_array = np.array(rmssd_all, dtype=np.float32)

    return rr_windows_array, rmssd_array, max_len_global


# Dataset
class RRDataset(Dataset):
    def __init__(self, rr_windows, rmssd_values):
        self.rr_windows = torch.tensor(rr_windows, dtype=torch.float32)
        self.rmssd_values = torch.tensor(rmssd_values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.rmssd_values)

    def __getitem__(self, idx):
        return self.rr_windows[idx], self.rmssd_values[idx]

fs = 130
window_sec = 10
step_sec = 2
batch_size = 32
epochs = 50
lr = 0.001

# 2. Przygotowanie danych
rr_train, rmssd_train, input_len_train = prepare_dataset_from_multiple_csvs(   ["data/R5.csv", "data/R7.csv", "data/R1.csv", "data/R8.csv"], fs, window_sec, step_sec)
rr_test, rmssd_test, input_len_test = prepare_dataset_from_multiple_csvs(   ["data/R4.csv","data/R5.csv"], window_sec, step_sec)

# Upewnij się, że długości wejść są takie same
input_len = max(input_len_train, input_len_test)
rr_train = np.pad(rr_train, ((0,0), (0, input_len - rr_train.shape[1])), 'constant')
rr_test = np.pad(rr_test, ((0,0), (0, input_len - rr_test.shape[1])), 'constant')

# 3. Konwersja na tensory
X_train = torch.tensor(rr_train, dtype=torch.float32)
y_train = torch.tensor(rmssd_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(rr_test, dtype=torch.float32)
y_test = torch.tensor(rmssd_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 4. Model
class RMSSDModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RMSSDModel(input_size=input_len)

# 5. Optymalizacja
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 6. Trenowanie
print("Trening...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 7. Testowanie
print("\nTestowanie...")
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()
    targets = y_test.squeeze().numpy()
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.2f} ms")
