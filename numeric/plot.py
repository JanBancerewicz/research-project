
import matplotlib.pyplot as plt
import numpy as np

from numeric.rr import filter_ecg_with_timestamps, find_r_peaks_values_with_timestamps, fft_rr_intervals


def plot(data):
    timestamps = data["timestamp"]  # 10 seconds of ECG signal sampled at 1000 Hz
    ecg_values = data["ecg"]  # Example synthetic ECG signal

    # Sampling frequency
    fs = 130  # 1000 Hz
    ecg_signal = np.column_stack((timestamps, ecg_values))

    f = filter_ecg_with_timestamps(ecg_signal, fs)

    # 2. Find R-peaks with timestamps
    r_peaks_with_timestamps = find_r_peaks_values_with_timestamps(f, fs)

    # 3. Plot the ECG signal with detected R-peaks

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, f[:,1], label="ECG Signal", color="b")



    plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:,1], "ro", label="Detected R-peaks")
    plt.title("ECG Signal with Detected R-peaks")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:, 1], label="R-peaks")
    plt.title("R")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    p = r_peaks_with_timestamps[:, 0]

    x = np.diff(p)

    # 4. Calculate FFT of RR intervals
    freqs, rr_fft = fft_rr_intervals(x, fs)
    plt.subplot(3, 1, 3)
    plt.title("R-R line")
    plt.plot(x, label="R-R")
    plt.legend()

    plt.tight_layout()

    heart_rate = 60000 / x
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

   # b = estimate_breath_rate(x)
   # print(f"avg breath: {b}")

    plt.show()
