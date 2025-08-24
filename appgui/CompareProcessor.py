import numpy as np

class CompareProcessor:

    def __init__(self):
        self.ppg_peaks = []  # List of times (seconds, int)
        self.ecg_peaks = []  # List of times (seconds, int)
        self.diff = []       # List to store differences between PPG and ECG peaks
        self.big_epsilon = 3000

    def add_ecg_peaks(self, arr):
        """Add detected ECG peak times (seconds)."""
        arr = np.array(arr, dtype=np.int64)  # castowanie do int
        self.ecg_peaks.extend(arr.tolist())
        self.compare()

    def add_ppg_peaks(self, arr):
        """Add detected PPG peak times (seconds)."""
        arr = np.array(arr, dtype=np.int64)  # castowanie do int
        print(f"PPG peak: {arr}")
        self.ppg_peaks.extend(arr.tolist())
        self.compare()

    def align_auto(self, s1, s2):
        """Synchronize two sorted timestamp lists and compute differences."""

        # decydujemy kto zaczyna wcześniej
        if s1[0] <= s2[0]:
            # znajdź start w s1 względem pierwszego s2
            first = s2[0]
            start_idx = 0
            while start_idx < len(s1) and s1[start_idx] <= first:
                start_idx += 1
            start_idx -= 1
            if start_idx < 0:
                return [], [], []
            i, j = start_idx, 0
        else:
            # znajdź start w s2 względem pierwszego s1
            first = s1[0]
            start_idx = 0
            while start_idx < len(s2) and s2[start_idx] <= first:
                start_idx += 1
            start_idx -= 1
            if start_idx < 0:
                return [], [], []
            i, j = 0, start_idx

        s1_aligned, s2_aligned, diffs = [], [], []
        while i < len(s1) and j < len(s2):
            if s1[i] <= s2[j]:
                s1_aligned.append(s1[i])
                s2_aligned.append(s2[j])
                diffs.append(s2[j] - s1[i])
                i += 1
                j += 1
            else:
                j += 1

        return s1_aligned, s2_aligned, diffs

    def compare(self):
        if not self.ecg_peaks or not self.ppg_peaks:
            return

        s1, s2, diffs = self.align_auto(self.ppg_peaks, self.ecg_peaks)
        self.diff = diffs

        if diffs:
            print(diffs)
            print("Peaks time diff (ECG - PPG): mean={:.4f}s std={:.4f}s".format(np.mean(diffs), np.std(diffs)))
        else:
            print("No peak diffs calculated.")
