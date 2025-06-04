import numpy as np
import pandas as pd
import neurokit2 as nk
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

from cnn.rmssd.RMSSDModel import RMSSDModel


class RRDataset(Dataset):
    def __init__(self, rr_windows, rmssd_values):
        self.rr_windows = torch.tensor(rr_windows, dtype=torch.float32)
        self.rmssd_values = torch.tensor(rmssd_values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.rmssd_values)

    def __getitem__(self, idx):
        return self.rr_windows[idx], self.rmssd_values[idx]

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for rr_batch, rmssd_batch in dataloader:
            rr_batch = rr_batch.to(device)
            rmssd_batch = rmssd_batch.to(device)

            # Forward pass
            outputs = model(rr_batch)
            loss = criterion(outputs, rmssd_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rr_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


def load_ecg_and_rpeaks(csv_path, sampling_rate):
    df = pd.read_csv(csv_path)
    ecg_signal = df['ecg'].values
    r_peaks = np.where(df['R'].values == 1)[0]  # Indices of R-peaks
    r_peaks = r_peaks / sampling_rate
    return ecg_signal, r_peaks

def compute_rr_intervals(r_peak_indices):
    rr_intervals = np.diff(r_peak_indices)
    return rr_intervals  * 1000

def compute_rmssd(rr_intervals):
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    return rmssd


def create_rr_windows(rr_intervals, window_size, step_size=1):
    """
    Splits RR intervals into overlapping (moving) windows with a fixed step size.

    Args:
        rr_intervals (np.array): RR interval array (ms).
        window_size (int): Number of RR intervals per window.
        step_size (int): Step size for moving window (default = 1).

    Returns:
        Tuple of:
            - windows (np.array): shape (N, window_size)
            - rmssd_values (np.array): shape (N,)
    """
    windows = []
    rmssd_values = []

    for i in range(0, len(rr_intervals) - window_size + 1, step_size):
        window = rr_intervals[i:i + window_size]
        rmssd = compute_rmssd(window)
        windows.append(window)
        rmssd_values.append(rmssd)

    return windows, rmssd_values


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for rr_batch, rmssd_batch in dataloader:
            rr_batch = rr_batch.to(device)
            rmssd_batch = rmssd_batch.to(device)

            outputs = model(rr_batch)
            loss = criterion(outputs, rmssd_batch)
            total_loss += loss.item() * rr_batch.size(0)

            predictions.append(outputs.cpu())
            targets.append(rmssd_batch.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    print(f"\nTest Loss: {avg_loss:.4f}")
    return predictions, targets


data = ['../../data/R4.csv', '../../data/R1.csv', '../../data/R2.csv', '../../data/R5.csv', '../../data/R6.csv',
        '../../data/r/R7.csv', '../../data/r/R11.csv', '../../data/r/R12.csv']

train_windows = []
train_rmssd_values = []

for d in data:
    ecg_signal, r_peaks = load_ecg_and_rpeaks(d, 130)
    rr_intervals = compute_rr_intervals(r_peaks)
    windows, values = create_rr_windows(rr_intervals, 10)

    train_windows.extend(windows)
    train_rmssd_values.extend(values)


sampling_rate = 130
window_size = 10

test_file = '../../data/R8.csv'
ecg_signal, r_peak_times = load_ecg_and_rpeaks(test_file, sampling_rate)
rr_intervals = compute_rr_intervals(r_peak_times)
test_windows, test_rmssd_values = create_rr_windows(rr_intervals, window_size)

test_windows = np.array(test_windows)
test_rmssd_values = np.array(test_rmssd_values)

# Create datasets and loaders
train_dataset = RRDataset(train_windows, train_rmssd_values)
test_dataset = RRDataset(test_windows, test_rmssd_values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RMSSDModel(input_size=window_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer, device, num_epochs=500)

# Evaluate the model
predictions, targets = evaluate(model, test_loader, criterion, device)


torch.save(model.state_dict(), "rmssd_model.pth")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(targets, label='True RMSSD')
plt.plot(predictions, label='Predicted RMSSD')
plt.title("RMSSD Prediction on Test Data")
plt.xlabel("Sample")
plt.ylabel("RMSSD (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmssd_combinded.png")


plt.figure()
plt.plot(predictions, label='Predicted RMSSD')
plt.title("RMSSD Prediction on Test Data")
plt.xlabel("Sample")
plt.ylabel("RMSSD (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmsd_prediction.png")

plt.figure()
plt.plot(targets, label='True RMSSD')
plt.title("RMSSD True on Test Data")
plt.xlabel("Sample")
plt.ylabel("RMSSD (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmsd_true.png")

