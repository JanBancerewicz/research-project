import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Load ECG and PPG data from CSV files ---
ecg_df = pd.read_csv('ecg_data.csv')
ppg_df = pd.read_csv('ppg_data.csv')

ecg_signal = ecg_df['ecg'].values[100:600]
ecg_time = ecg_df['time'].values[100:600]
ppg_signal = ppg_df['ppg'].values[400:800]
ppg_time = ppg_df['time'].values[400:800]

# --- Filter parameters ---
fs_ecg = 130  # Hz, adjust if needed
fs_ppg = 30   # Hz, adjust if needed
ecg_lowcut, ecg_highcut = 0.5, 45
ppg_lowcut, ppg_highcut = 0.5, 8

# --- Filter signals ---
ecg_filtered = bandpass_filter(ecg_signal, fs_ecg, ecg_lowcut, ecg_highcut)
ppg_filtered = bandpass_filter(ppg_signal, fs_ppg, ppg_lowcut, ppg_highcut)

# --- Scale PPG to [-1, 1] in moving windows of 100 samples ---
def scale_to_unit_windowed(signal, window=100):
    scaled = np.zeros_like(signal)
    n = len(signal)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        window_slice = signal[start:end]
        min_val = np.min(window_slice)
        max_val = np.max(window_slice)
        if max_val - min_val == 0:
            scaled[i] = 0
        else:
            scaled[i] = 2 * (signal[i] - min_val) / (max_val - min_val) - 1
    return scaled

ppg_signal_scaled = scale_to_unit_windowed(ppg_signal, window=100)
ppg_filtered_scaled = scale_to_unit_windowed(ppg_filtered, window=100)

# --- Plot ECG ---
plt.figure(figsize=(12, 5))
plt.plot(ecg_time, ecg_signal, label='ECG Raw', alpha=0.7)
plt.plot(ecg_time, ecg_filtered, label='ECG Filtered', alpha=0.7)
plt.title('ECG Signal (Raw and Filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig("Filtered_ecg.png", dpi=300)


# --- Plot PPG (scaled) ---
plt.figure(figsize=(12, 5))
plt.plot(ppg_time, ppg_signal_scaled, label='PPG Raw (scaled)', alpha=0.7)
plt.plot(ppg_time, ppg_filtered_scaled, label='PPG Filtered (scaled)', alpha=0.7)
plt.title('PPG Signal (Raw and Filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude (scaled)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Filtered_ppg.png", dpi=300)
plt.show()
