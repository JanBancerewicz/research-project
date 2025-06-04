import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.signal import butter, filtfilt


def get_hrv(r):
    rr = np.diff(r)
    rr_times = np.cumsum(rr)

    # Interpolate to get evenly sampled RR signal
    fs = 4  # Hz - typical for breathing rate tracking
    time_interp = np.arange(0, rr_times[-1], 1 / fs)
    rr_interp = np.interp(time_interp, rr_times, rr)

    # Bandpass filter in respiratory frequency range (~0.1–0.4 Hz)
    b, a = butter(N=2, Wn=[0.1, 0.4], btype='bandpass', fs=fs)
    rsp_est = filtfilt(b, a, rr_interp)

    slope = np.diff(rsp_est, append=rsp_est[-1])
    phase = np.where(slope >= 0, 0, 1)  # 0 = Inhale, 1 = Exhale

    # ----- Plot with full background colored -----
    plt.figure(figsize=(12, 4))
    start = 0
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:  # phase change
            plt.axvspan(time_interp[start], time_interp[i],
                        color='lightblue' if phase[start] == 0 else 'mistyrose', alpha=0.3)
            start = i
    else:
        # color the last segment
        plt.axvspan(time_interp[start], time_interp[-1],
                    color='lightblue' if phase[start] == 0 else 'mistyrose', alpha=0.3)

    # Plot respiration line
    plt.plot(time_interp, rsp_est, color='black', label='Estimated Respiration')
    plt.title("Inhale (blue) and Exhale (red) Phases from HRV")
    plt.xlabel("Time (s)")
    plt.ylabel("Respiration Signal")
    plt.legend()
    plt.tight_layout()


p = pd.read_csv("data/r/R21.csv")
r = []
for i in range(len(p["R"])):
    if p["R"][i] != 0:
        r.append(i * (1 / 130))

get_hrv(r[:200])

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Your RMSSD-based phase detection code
def compute_rmssd(rr_window):
    diff = np.diff(rr_window)
    return np.sqrt(np.mean(diff ** 2))


def sliding_rmssd(rr_intervals, win_size=5.0, step_size=1.0):
    rr_times = np.cumsum(rr_intervals)
    total_time = rr_times[-1]

    times = []
    rmssd_vals = []

    start = 0.0
    while start + win_size <= total_time:
        end = start + win_size
        mask = (rr_times >= start) & (rr_times < end)
        rr_win = rr_intervals[mask]
        if len(rr_win) >= 3:
            rmssd = compute_rmssd(rr_win)
            rmssd_vals.append(rmssd)
            times.append(start + win_size / 2)
        start += step_size

    return np.array(times), np.array(rmssd_vals)


def rmssd_resp_phase(rr_intervals, fs_interp=4.0):
    times, rmssd_vals = sliding_rmssd(rr_intervals)

    if len(times) < 2:
        return None, None, None

    t_interp = np.arange(0, times[-1], 1 / fs_interp)
    rmssd_interp = np.interp(t_interp, times, rmssd_vals)

    b, a = butter(N=2, Wn=[0.1, 0.4], btype='bandpass', fs=fs_interp)
    rmssd_filt = filtfilt(b, a, rmssd_interp)

    slope = np.diff(rmssd_filt, append=rmssd_filt[-1])
    phase = np.where(slope >= 0, 0, 1)

    return t_interp, rmssd_filt, phase


# Example RR data (simulate ~1 Hz breathing modulated RRIs)
t = np.linspace(0, 60, 300)  # 60s of data, 300 points
rr = 1.0 + 0.05 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(len(t))  # simulated RRIs
rr = np.diff(r[:200])
# Get RMSSD-based respiratory phase
times, rmssd_filtered, phase = rmssd_resp_phase(rr)

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(times, rmssd_filtered, label='Filtered RMSSD (0.1–0.4 Hz)', color='blue')
plt.fill_between(times, rmssd_filtered.min(), rmssd_filtered.max(), where=phase == 0, color='green', alpha=0.2,
                 label='Inhale')
plt.fill_between(times, rmssd_filtered.min(), rmssd_filtered.max(), where=phase == 1, color='red', alpha=0.2,
                 label='Exhale')
plt.title("RMSSD-based Respiratory Phase Detection")
plt.xlabel("Time (s)")
plt.ylabel("Filtered RMSSD")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Step 1: Preprocessing functions (R-peaks to RR intervals, RMSSD calculation)
def rpeak_to_rr_intervals(r_peaks):
    """
    Given the list of R-peaks (in time or sample indices), returns the RR intervals.
    """
    rr_intervals = np.diff(r_peaks)
    return rr_intervals


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Function to compute RMSSD for RR intervals
def compute_rmssd(rr_intervals):
    """
    Compute RMSSD (Root Mean Square of Successive Differences).
    """
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff ** 2))


# Function to create sliding windows for RMSSD calculation
def sliding_rmssd_s(rr_intervals, win_size=5.0, step_size=1.0):
    rr_times = np.cumsum(rr_intervals)
    total_time = rr_times[-1]

    times = []
    rmssd_vals = []

    start = 0.0
    while start + win_size <= total_time:
        end = start + win_size
        mask = (rr_times >= start) & (rr_times < end)
        rr_win = rr_intervals[mask]

        if len(rr_win) >= 3:
            rmssd = compute_rmssd(rr_win)
            rmssd_vals.append(rmssd)
            times.append(start + win_size / 2)

        start += step_size

    return np.array(times), np.array(rmssd_vals)


# Define the Model (Feedforward Neural Network)
class RRIntervalRMSSDModel(nn.Module):
    def __init__(self, input_size):
        super(RRIntervalRMSSDModel, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# Prepare the data (convert RR intervals to features and RMSSD labels)
def prepare_data(r_peaks, window_size=20):
    rr_intervals = np.diff(r_peaks)  # Calculate RR intervals
    times, rmssd_vals = sliding_rmssd_s(rr_intervals, win_size=window_size)

    X = []
    y = []

    for i in range(window_size, len(rmssd_vals) - window_size):
        segment = rr_intervals[i - window_size:i + window_size]
        X.append(segment)
        y.append(rmssd_vals[i])

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Test the model (check shape of input and output)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch


def test_model(model, rr_intervals, r, window_size=3):
    model.eval()

    # Compute RMSSD for sliding windows (ground truth)
    _, rmssd_vals = sliding_rmssd_s(rr_intervals, win_size=window_size)

    # Prepare the test data
    X_test = []
    y_true = []
    for i in range(window_size, len(rmssd_vals) - window_size):
        segment = rr_intervals[i - window_size:i + window_size]
        X_test.append(segment)
        y_true.append(rmssd_vals[i])

    # Ensure that X_test is not empty
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    if X_test.shape[0] == 0:
        print("Warning: X_test is empty. Reducing the window size or increasing the data may help.")
        return

    # Convert to tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Run the model to get predictions
    with torch.no_grad():
        preds = model(X_test_tensor).squeeze().numpy()

    # Print shapes
    print(f"Input to model (X_test) shape: {X_test_tensor.shape}")
    print(f"Predictions shape: {preds.shape}")

    # Evaluation metrics
    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    print(f"\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Visualization: Plot predicted vs true RMSSD
    t_plot = np.arange(len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(t_plot, preds, label="Predicted RMSSD", color='blue')
    plt.plot(t_plot, y_true, label="True RMSSD", color='green', linestyle='dashed')
    plt.title("Predicted vs True RMSSD")
    plt.xlabel("Window index")
    plt.ylabel("RMSSD")
    plt.legend()
    plt.tight_layout()
    plt.show()


import torch
import numpy as np


# --- Combine data from multiple RR signals ---
def prepare_multi_signal_data(rr_signal_list, window_size=10):
    all_X, all_y = [], []

    for rr_intervals in rr_signal_list:
        rr_intervals = np.array(np.diff(rr_intervals))
        times, rmssd_vals = sliding_rmssd_s(rr_intervals, win_size=window_size)

        # Build X, y windows
        for i in range(window_size, len(rmssd_vals) - window_size):
            segment = rr_intervals[i - window_size:i + window_size]
            all_X.append(segment)
            all_y.append(rmssd_vals[i])

    # Convert to tensors
    X = torch.tensor(np.array(all_X), dtype=torch.float32)
    y = torch.tensor(np.array(all_y), dtype=torch.float32)
    return X, y


# --- Example training loop ---
def train_model_on_multiple_signals(rr_signal_list, window_size=10, epochs=50):
    # Prepare training data
    X_train, y_train = prepare_multi_signal_data(rr_signal_list, window_size=window_size)

    print(f"Training on {len(X_train)} samples from {len(rr_signal_list)} signals.")

    # Create model
    model = RRIntervalRMSSDModel(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


# Main function to run the complete pipeline
if __name__ == "__main__":
    # Example R-peaks data (replace with your actual R-peak times)
    p = pd.read_csv("data/r/R20.csv")
    r2 = []
    r_peaks = []
    for i in range(len(p["R"])):
        if p["R"][i] != 0:
            r2.append(i * (1 / 130))
    r_peaks.append(r2)
    r2 = []
    p = pd.read_csv("data/r/R14.csv")
    for i in range(len(p["R"])):
        if p["R"][i] != 0:
            r2.append(i * (1 / 130))
    r_peaks.append(r2)
    r2 = []
    p = pd.read_csv("data/r/R13.csv")
    for i in range(len(p["R"])):
        if p["R"][i] != 0:
            r2.append(i * (1 / 130))
    r_peaks.append(r2)
    r2 = []
    p = pd.read_csv("data/r/R12.csv")
    for i in range(len(p["R"])):
        if p["R"][i] != 0:
            r2.append(i * (1 / 130))
    r_peaks.append(r2)
    # Prepare training data

    # Initialize and train the model
    model = train_model_on_multiple_signals(r_peaks, window_size=10)
    test_model(model, np.diff(r[:200]), r[:200], window_size=10)
