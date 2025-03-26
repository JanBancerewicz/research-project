import math

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd


data = pd.read_csv("ecg_data1.csv")

f = data["Timestamp"][0]
d = data["Timestamp"][len(data["Timestamp"])-1]

print(math.floor((d-f)))
print(len(data["Timestamp"])/130)

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d


# PLIK DO ROZPOZNAWANIA SZCZYTOW W SYGNALE

def bandpass_filter(sig, frequency, lowcut: float = 5, highcut: float = 18):
    nyquist = 0.5 * frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def filter_ecg_with_timestamps(ecg_signal, frequency: float):
    timestamps = ecg_signal[:, 0]
    ecg_values = ecg_signal[:, 1]

    filtered_signal = bandpass_filter(ecg_values, frequency, lowcut=0.5, highcut=40)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.150 * frequency)  # 150 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    # Combine timestamps with filtered signal values
    filtered_signal_with_timestamps = np.column_stack((timestamps, integrated_signal))

    return filtered_signal_with_timestamps


def derivative_filter(sig):
    return np.diff(sig, prepend=0)


def square(sig):
    return sig ** 2


def moving_window_integration(sig, window_size):
    return np.convolve(sig, np.ones(window_size), mode="same")


def refine_peak_positions(ecg_signal, detected_peaks, search_window=10):
    refined_peaks = []

    for peak in detected_peaks:
        start = max(peak - search_window, 0)  # Ensure the window doesn't go out of bounds
        end = min(peak + search_window, len(ecg_signal) - 1)

        refined_peak = np.argmax(ecg_signal[start:end]) + start
        refined_peaks.append(refined_peak)

    return np.array(refined_peaks)


def find_r_peaks_values(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal, frequency)

    return ecg_signal[peaks, 1]


def find_r_peaks_values_with_timestamps(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal[:, 1], frequency)

    peak_values = ecg_signal[peaks, 1]
    peak_timestamps = ecg_signal[peaks, 0]

    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))

    return peaks_with_timestamps


def find_r_peaks_ind(ecg_signal, frequency: float):
    filtered_signal = bandpass_filter(ecg_signal, frequency)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.050 * frequency)  # 50 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    clipped_signal = np.clip(integrated_signal, 0, np.percentile(integrated_signal, 99))
    threshold = np.mean(clipped_signal) + 0.6 * np.std(clipped_signal)
    peaks, _ = signal.find_peaks(
        integrated_signal, height=threshold, distance=int(0.4 * frequency)  # 400 ms
    )

    refined_peaks = refine_peak_positions(ecg_signal, peaks, round(10 / 130 * frequency))
    # refined_peaks = processing.correct_peaks(
    #     sig=ecg_signal,
    #     peak_inds=peaks,
    #     search_radius=30,
    #     smooth_window_size=30,
    #     # peak_dir="up",
    # )
    # refined_peaks = peaks

    return refined_peaks

# Calculate the FFT of RR intervals
def fft_rr_intervals(rr_intervals, fs):
    # Apply FFT to the RR intervals
    rr_fft = np.fft.fft(rr_intervals)
    # Calculate the corresponding frequency axis
    freqs = np.fft.fftfreq(len(rr_intervals), d=1/fs)

    # Get the positive frequencies (we usually ignore the negative frequencies)
    positive_freqs = freqs[:len(freqs)//2]
    positive_rr_fft = np.abs(rr_fft)[:len(rr_fft)//2]  # Use absolute value of FFT results

    return positive_freqs, positive_rr_fft

timestamps = data["Timestamp"]  # 10 seconds of ECG signal sampled at 1000 Hz
ecg_values = data["ECG"]  # Example synthetic ECG signal

# Sampling frequency
fs = 130  # 1000 Hz
ecg_signal = np.column_stack((timestamps, ecg_values))

# 1. Filter the ECG with timestamps
filtered_signal_with_timestamps = filter_ecg_with_timestamps(ecg_signal, fs)

# 2. Find R-peaks with timestamps
r_peaks_with_timestamps = find_r_peaks_values_with_timestamps(ecg_signal, fs)

# 3. Plot the ECG signal with detected R-peaks

plt.figure(figsize=(10, 6))
plt.plot(timestamps, ecg_values, label="ECG Signal", color="b")
plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:, 1], "ro", label="Detected R-peaks")
plt.title("ECG Signal with Detected R-peaks")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:, 1],  label="R-peaks")
plt.title("R")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()



p = r_peaks_with_timestamps[:, 0]

x = np.diff(p) / fs / 10000000




# 4. Calculate FFT of RR intervals
freqs, rr_fft = fft_rr_intervals(x, fs)
plt.subplot(3, 1, 3)
plt.title("R-R line")
plt.plot(x, label="R-R")
plt.legend()



plt.tight_layout()

heart_rate = 60 / x
plt.figure(figsize=(10, 6))
plt.plot(heart_rate, label="HR")
plt.title("Heart Rate")
plt.xlabel("Time [s]")
plt.ylabel("bmp")
plt.legend()

# 5. Plot the FFT of RR intervals
plt.figure(figsize=(10, 6))
plt.plot(freqs, rr_fft, label="FFT of RR intervals")
plt.title("FFT of RR Intervals")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.legend()




plt.show()
