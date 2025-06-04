import os
import re

import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

lowcut = 0.1
highcut = 0.5
fs = 130

def get_r_amplitudes(r, ecg, timestamps):
    r_amplitudes = []
    r_timestamps = []
    for i in range(len(r)):
        if r[i] == 1:
            r_amplitudes.append(ecg[i])
            r_timestamps.append(timestamps[i])
    return r_amplitudes, r_timestamps

def get_minute_range(prev, timestamps):
    for i in range(prev, len(timestamps)):
        if timestamps[i] - timestamps[prev] >= 60:
            return i + 1
    return -1

def get_timestamp_start_end(start, end, timestamps):
    s = 0
    e = 0
    for i in range(len(timestamps)):
        if timestamps[i]  == start:
            s = i
        if timestamps[i] == end:
            e = i
            break
    return s, e

def get_edg(file, file_r):
    data = pd.read_csv(file)
    ecg = data["ecg"]
    timestamps = data["timestamp"] / 1000

    r_data = pd.read_csv(file_r)
    r_peaks = r_data["R"]

    r_amplitudes, time_r = get_r_amplitudes(r_peaks, ecg, timestamps)



    # Zmienna do przechowywania wyników
    b = []

    # Inicjalizacja indeksu startowego

    i = 0
    prev = 0
    chunks = len(time_r) // 100
    idx_c = 0
    while idx_c < chunks:
        idx_c += 1
        end = 100 + prev
        if end > len(r_peaks):
            break


        edr_raw = r_amplitudes[prev:end]
        edr_smooth = savgol_filter(edr_raw, window_length=11, polyorder=2)
        edr_interp_func = interp1d(time_r[prev:end], edr_smooth, kind='cubic', fill_value="extrapolate")
        s, e = get_timestamp_start_end(time_r[prev], time_r[end-1], timestamps)
        edr_full = edr_interp_func(timestamps[s:e])

        print(f"chunks: {idx_c}/{chunks}")

        start_index = prev
        end_index = end
        prev = end
        segment_edr = edr_full

        # Detekcja szczytów R w segmencie EDR
        peaks, _ = find_peaks(segment_edr, distance=int(fs * 2))  # Oddechy co ≥2 sekundy

        # Liczenie oddechów na minutę
        num_breaths = len(peaks)
        duration_sec = time_r[end_index - 1] - time_r[start_index]  # Użyj ostatniego dostępnego indeksu
        breaths_per_min = (num_breaths / duration_sec) * 60
        b.append((idx_c, breaths_per_min))

    print(b)
    return b


def get_r_file(ecg_file):
    match = re.search(r'\d+', ecg_file)

    if match:
        number = int(match.group())
        return f"data/r/R{number}.csv", number
    return ""


def process_edg():
    folder_path = "data/ecg"

    ecg_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for ecg_file in ecg_files:
        print(f"processing {ecg_file}")
        r_file, num = get_r_file(ecg_file)
        breath = get_edg("data/ecg/"+ecg_file, r_file)
        df = pd.DataFrame(breath, columns=["chunk","breath"])
        df.to_csv(f"data/breath/B{num}.csv", index=False)

process_edg()



#