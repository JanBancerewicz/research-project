import random

import numpy as np


class CompareProcessor:

    def __init__(self):
        self.ppg_peaks = []  # List of times (seconds)
        self.ecg_peaks = []  # List of times (seconds)
        self.diff = []  # List to store differences between PPG and ECG peaks
        self.big_epsilon = 3000

    def add_ecg_peaks(self, arr):
        """Add detected ECG peak times (seconds)."""
        self.ecg_peaks.extend(arr)
        self.compare()

    def add_ppg_peaks(self, arr):
        """Add detected PPG peak times (seconds)."""
        print(f"PPG peak: {arr}")
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
        start_index = 0
        if len(ecg_times) < len(ppg_times):
            for i in range(len(ppg_times)):
                if abs(ecg_times[i] - ppg_times[i]) > self.big_epsilon:
                    start_index = i
                    break
            for i in range(start_index, len(ppg_times)):
                if i < len(ecg_times):
                    diff = ppg_times[i] - ecg_times[i]
                    diffs.append(diff)
        else:
            for i in range(len(ecg_times)):
                if abs(ppg_times[i] - ecg_times[i]) > self.big_epsilon:
                    start_index = i
                    break
            for i in range(start_index, len(ecg_times)):
                if i < len(ppg_times):
                    diff = ppg_times[i] - ecg_times[i]
                    diffs.append(diff)

        self.diff = diffs

        if diffs:
            print(diffs)
            print("Peaks time diff (PPG - ECG): mean={:.4f}s std={:.4f}s".format(np.mean(diffs), np.std(diffs)))
        else:
            print("No peak diffs calculated.")
