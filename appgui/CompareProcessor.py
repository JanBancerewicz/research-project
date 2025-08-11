import random

import numpy as np


class CompareProcessor:

    def __init__(self):
        self.ppg_peaks = []  # List of times (seconds)
        self.ecg_peaks = []  # List of times (seconds)
        self.diff = []  # List to store differences between PPG and ECG peaks

    def add_ecg_peaks(self, arr):
        """Add detected ECG peak times (seconds)."""
        self.ecg_peaks.extend(arr)
        a = []
        for i in range(max(len(arr)-1, 1)):
            a.append(arr[i] +  random.randint(0,1000)/1000.0)  # Simulate some noise in ECG peaks
        self.ppg_peaks.extend(a)
        self.compare()

    def add_ppg_peaks(self, arr):
        """Add detected PPG peak times (seconds)."""
        self.ppg_peaks.extend(arr)
        self.compare()

    def compare(self):
        # Only compare if both lists have at least one peak
        if not self.ecg_peaks or not self.ppg_peaks:
            return

        ecg_times = self.ecg_peaks
        ppg_times = self.ppg_peaks

        # For each PPG peak, find the closest ECG peak in time
        diffs = []
        for ppg_time in ppg_times:
            idx = np.argmin(np.abs(np.array(ecg_times) - ppg_time))
            diff = ppg_time - ecg_times[idx]
            diffs.append(diff)

        self.diff = diffs

        if diffs:
            print(diffs)
            print("Peaks time diff (PPG - ECG): mean={:.4f}s std={:.4f}s".format(np.mean(diffs), np.std(diffs)))
        else:
            print("No peak diffs calculated.")
