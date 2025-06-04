import numpy as np
import pandas as pd

from numeric.rr import find_r_peaks_values_with_timestamps, filter_ecg_with_timestamps


def extract_r_indexes(data):
    timestamps = data["timestamp"]
    ecg_values = data["ecg"]  # Example synthetic ECG signal

    fs = 130
    ecg_signal = np.column_stack((timestamps, ecg_values))
    #f = filter_ecg_with_timestamps(ecg_signal, fs)

    r_peaks_with_timestamps = find_r_peaks_values_with_timestamps(ecg_signal, fs)

    r_indexes = []

    N = len(ecg_signal[:, 0])
    y = 0
    for x in range(N):
        if y != len(r_peaks_with_timestamps[:, 0]):
            if r_peaks_with_timestamps[y, 0] == ecg_signal[x, 0]:
                r_indexes.append((ecg_signal[x, 1],1))
                y += 1
            else:
                r_indexes.append((ecg_signal[x, 1], 0))
        else:
            r_indexes.append((ecg_signal[x, 1], 0))
    return r_indexes


def save_csv(r_indexes, file):
    df = pd.DataFrame(r_indexes, columns=["ecg", "R"])
    df.to_csv(file, index=False)


