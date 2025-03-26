import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft, fftfreq


def get_peaks_and_valleys(heart_rate):
    peaks, _ = find_peaks(heart_rate, height=30, prominence=0.5)  # Customize conditions as needed

    inverted_y = -heart_rate

    valleys, _ = find_peaks(inverted_y)

    return peaks, valleys

def get_peak_diff(peaks, valleys, heart_rate):
    peaks_rate = []
    valleys_rate = []
    for val in valleys:
        valleys_rate.append(heart_rate[val])
    for peak in peaks:
        peaks_rate.append(heart_rate[peak])

    peaks_diff = []
    if peaks[0] < valleys[0]:
        for i in range(len(valleys_rate)):
            peaks_diff.append(peaks_rate[i] - valleys_rate[i])
            try:
                peaks_diff.append(valleys_rate[i] - peaks_rate[i + 1])
            except IndexError:  # KIEP detected
                pass
    else:
        for i in range(len(peaks_rate)):
            peaks_diff.append(peaks_rate[i] - valleys_rate[i])
            try:
                peaks_diff.append(valleys_rate[i + 1] - peaks_rate[i])
            except IndexError:
                pass
    return peaks_diff

def plot_heart_rate(peaks, valleys, heart_rate):
    plt.figure(1)
    plt.plot(heart_rate, label='Heart Rate Data')
    #plt.plot(peaks, heart_rate[peaks], "x", label='Detected Peaks', color='red')
    #plt.plot(valleys, heart_rate[valleys], "x", label='Detected valleys', color='green')
    plt.xlabel('Time (Index)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Heart Rate Data with Detected Peaks')
    plt.legend()
    plt.grid(True)


def plot_peaks_diff(peaks_diff):
    plt.figure(2)
    plt.plot(np.arange(len(peaks_diff)), np.array(peaks_diff), label='Peaks diff')
    plt.xlabel('Time (Index)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Peaks diff')
    plt.legend()
    plt.grid(True)


def plot_breath(heart_rate):



    # Extract heart rate and time

    time = np.arange(len(heart_rate))


    n = len(heart_rate)
    dt = np.mean(np.diff(time))

    frequencies = fftfreq(n, d=dt)
    fft_values = fft(heart_rate)  # Compute FFT

    resp_band = (frequencies > 0.1) & (frequencies < 0.5)
    filtered_fft = np.zeros_like(fft_values)
    filtered_fft[resp_band] = fft_values[resp_band]

    respiratory_signal = np.real(ifft(filtered_fft))

    exhale = np.diff(respiratory_signal) > 0
    inhale = ~exhale



    plt.figure(figsize=(6, 10))

    # Original Heart Rate Signal
    plt.subplot(2, 1, 1)
    plt.plot(time, heart_rate, label="Heart Rate (BPM)", color="gray", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Original Heart Rate Signal")
    plt.legend()

    # Respiratory Signal with Inhalation & Exhalation Phases
    plt.subplot(2, 1, 2)
    plt.plot(time[1:], respiratory_signal[1:], label="Respiratory Signal", color="black", alpha=0.7)
    plt.scatter(time[1:][inhale], respiratory_signal[1:][inhale], color='red', label="Inhalation")
    plt.scatter(time[1:][exhale], respiratory_signal[1:][exhale], color='blue', label="Exhalation")
    plt.xlabel("Time (s)")
    plt.ylabel("Respiratory Signal")
    plt.title("Breathing Phases from Heart Rate")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, heart_rate, label="Heart Rate (BPM)", color="gray", alpha=0.5)

    plt.scatter(time[1:][inhale], heart_rate[1:][inhale], color='red', label="Inhalation")
    plt.scatter(time[1:][exhale], heart_rate[1:][exhale], color='blue', label="Exhalation")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Breathing Phases from Heart Rate & Heat Rate")
    plt.legend()

    plt.tight_layout()


def plot_all(file):
    df = pd.read_csv(file)
    heart_rate = df['Heart Rate']

    peaks, valleys = get_peaks_and_valleys(heart_rate)



    peaks_diff = get_peak_diff(peaks, valleys, heart_rate)


    plot_heart_rate(peaks, valleys, heart_rate)
    plot_peaks_diff(peaks_diff)
    plt.tight_layout()
    plt.show()
    #plot_breath(heart_rate)


