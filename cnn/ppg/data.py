import neurokit2 as nk
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from cnn.ppg.PPGPeakDetector import PPGPeakDetector
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks


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


def generate_ppg_to_file(
    output_file="ppg_data_gen.csv",
    duration=1000,
    sampling_rate=30,
    lowcut=0.4,
    highcut=5
):
    # 1. Simulate PPG
    raw_ppg = nk.ppg_simulate(duration=duration, sampling_rate=sampling_rate)

    # 2. Apply bandpass filter
    filtered_ppg = butter_bandpass_filter(raw_ppg, lowcut, highcut, fs=sampling_rate)

    # 3. Normalize
    normalized_ppg = normalize_signal(filtered_ppg)

    # 4. Detect peaks using scipy
    peak_indices, _ = find_peaks(normalized_ppg, distance=sampling_rate//2, prominence=0.1)
    peaks = np.zeros_like(normalized_ppg)
    peaks[peak_indices] = 1

    # 5. Save to CSV
    df = pd.DataFrame({
        "ppg": normalized_ppg,
        "peak": peaks
    })
    plt.plot(normalized_ppg)
    plt.show()
    df.to_csv(output_file, index=False)
    print(f"✅ PPG data saved to {output_file}")





def generate_ppg_segment(segment_length=50, sampling_rate=100):
    signal = nk.ppg_simulate(duration=10, sampling_rate=sampling_rate)
    signals, info = nk.ppg_process(signal, sampling_rate=sampling_rate)
    ppg_clean = signals["PPG_Clean"].values
    peaks = signals["PPG_Peaks"].values.astype(int)

    plt.plot(ppg_clean)

    segments = []
    labels = []

    for i in range(len(ppg_clean) - segment_length):
        segment = ppg_clean[i:i + segment_length]
        label = peaks[i:i + segment_length]
        segments.append(segment)
        labels.append(label)

    return np.array(segments), np.array(labels)



def load_ppg_segments_from_csv(filepath, segment_length=50):
    df = pd.read_csv(filepath)
    ppg = df['ppg'].values
    peaks = df['peak'].values.astype(int)

    segments = []
    labels = []

    # Create overlapping segments (stride=1)
    for i in range(len(ppg) - segment_length):
        segment = ppg[i:i+segment_length]
        label = peaks[i:i+segment_length]

        segments.append(segment)
        labels.append(label)

    return np.array(segments), np.array(labels)


import torch
from torch.utils.data import Dataset

class PPGFileDataset(Dataset):
    def __init__(self, csv_path, max_segments=None):
        X, y = load_ppg_segments_from_csv(csv_path)
        if max_segments:
            X = X[:max_segments]
            y = y[:max_segments]

        # Already normalized between -1 and 1 from generation step
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, 50)
        self.y = torch.tensor(y, dtype=torch.float32)               # shape: (N, 50)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



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
            output = model(X_batch)         # Now model and data are on same device
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")



import matplotlib.pyplot as plt


def test_model(model, dataset, num_windows=100):
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_ppg = []
    all_true = []
    all_pred = []

    for i in range(num_windows):
        X_sample, y_true = dataset[i]  # (1, 50)
        X_tensor = X_sample.unsqueeze(0).to(device)  # (1, 1, 50)

        with torch.no_grad():
            y_pred = model(X_tensor).squeeze().cpu().numpy()

        all_ppg.extend(X_sample.squeeze().numpy())  # raw signal
        all_true.extend(y_true.numpy())
        all_pred.extend((y_pred > 0.5).astype(int))

    all_ppg = np.array(all_ppg)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    true_peak_indices = np.where(all_true == 1)[0]
    pred_peak_indices = np.where(all_pred == 1)[0]

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(all_ppg, color='black', label='PPG Signal')
    plt.scatter(true_peak_indices, all_ppg[true_peak_indices], color='green', label='True Peaks', zorder=5)
    plt.scatter(pred_peak_indices, all_ppg[pred_peak_indices], color='red', marker='x', label='Predicted Peaks', zorder=5)
    plt.title(f"PPG Peak Detection over {num_windows} Windows")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Prepare data
dataset = PPGFileDataset("ppg_data_gen.csv", max_segments=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize and train model
model = PPGPeakDetector()
train_model(model, dataloader, epochs=10)

# Test on one example
test_model(model, dataset)

