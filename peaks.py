import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
    plt.plot(peaks, heart_rate[peaks], "x", label='Detected Peaks', color='red')
    plt.plot(valleys, heart_rate[valleys], "x", label='Detected valleys', color='green')
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


def plot_all(file):
    df = pd.read_csv(file)
    heart_rate = df['Heart Rate']

    peaks, valleys = get_peaks_and_valleys(heart_rate)



    peaks_diff = get_peak_diff(peaks, valleys, heart_rate)


    plot_heart_rate(peaks, valleys, heart_rate)
    plot_peaks_diff(peaks_diff)

    plt.show()


