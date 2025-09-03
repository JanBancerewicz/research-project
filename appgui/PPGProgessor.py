from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter, find_peaks, filtfilt, butter
from cnn.ppg.data import get_or_train_model, predict_ppg_segment

import neurokit2 as nk
@dataclass
class PPGResult:
    time_array: np.ndarray
    filtered_signal: np.ndarray
    raw_signal: np.ndarray
    peak_times: list
    peak_values: list
    peak_unix_times: list


class PPGProcessor:
    def __init__(self, window_size=100, polyorder=3, peak_distance=3, peak_prominence=0.3):
        """
        window_size: number of samples per window (must be odd for savgol)
        peak_distance: minimal distance between peaks (in samples)
        peak_prominence: minimum prominence to be considered a peak
        """
        self.polyorder = polyorder
        self.window_size = window_size
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.time_unix = []



        self.sample_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)

        DATA_DIR = "cnn/ppg/train_data"
        MODEL_PATH = "ppg_peak_model.pth"
        SEGMENT_LENGTH = window_size
        MAX_SEGMENTS = 10000
        EPOCHS = 200
        BATCH_SIZE = 32
        LR = 0.001
        MAX_FILES = None

        self.model = get_or_train_model(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            segment_length=SEGMENT_LENGTH,
            max_segments=MAX_SEGMENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_files=MAX_FILES
        )

        self.r = [] # Store detected peak times


    def add_sample(self, sample, time, time_unix):
        """
        Add new PPG sample and time.
        When buffer is full, return a PPGResult.
        Otherwise, return None.
        """
        self.sample_buffer.append(sample)
        self.time_buffer.append(time)
        self.time_unix.append(time_unix)

        if len(self.sample_buffer) == self.window_size:
            # Convert buffers to numpy arrays
            window_data = np.array(self.sample_buffer)
            time_data = np.array(self.time_buffer)
            unix_data = np.array(self.time_unix)

            # Process the signal (filtering and normalization)
            filtered = self.process_func(window_data)

            # Detect peaks
            peak_times, peak_values = self.detect_peaks(filtered, time_data)

            # Match peak_times to unix times
            peak_unix_times = []
            for pt in peak_times:
                idx = np.where(time_data == pt)[0]
                if len(idx) > 0:
                    peak_unix_times.append(unix_data[idx[0]])
                else:
                    peak_unix_times.append(None)

            # Save R-peak times and calculate HRV
            for peak_time in peak_times:
                self.r.append(peak_time)
                # print(f"[DEBUG] Updated R-peaks (self.r): {self.r}")
                if len(self.r) >= 2:
                    # Calculate HRV (time difference between last two peaks)
                    hrv = (self.r[-1] - self.r[-2]) * 1000  # Convert to milliseconds
                    print(f"HRV (ms): {hrv:.2f}")

            # Calculate HRV
            hrv = self.compute_hrv()
            print(f"HRV Metrics: {hrv}")  # Log HRV metrics

            # Clear buffers after processing
            self.sample_buffer.clear()
            self.time_buffer.clear()
            self.time_unix.clear()

            # Return the result with unix peak times
            return PPGResult(
                time_array=time_data,
                filtered_signal=filtered,
                raw_signal=window_data,
                peak_times=peak_times,
                peak_unix_times=peak_unix_times,
                peak_values=peak_values
            ), hrv
        else:
            return None

    def bandpass_filter(self, signal_data, fs=30, lowcut=0.5, highcut=5.0, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal_data)

    def process_func(self, window_data):
        """
        Apply Savitzky-Golay filter.
        """
        return self._normalize_window(self.bandpass_filter(window_data))

    def detect_peaks(self, signal, time_array):
        """
        Detect peaks using neurokit2's PPG peak detection.
        Returns peak times and values.
        """
        try:
            out = predict_ppg_segment(self.model, signal)
            peak_times = []
            peak_values = []
            for i in range(len(out)):
                if out[i]:
                    print("------------", time_array[i], signal[i])
                    peak_times.append(time_array[i])
                    peak_values.append(signal[i])

            if len(peak_times) == 0:
                print("[PPGProcessor] No peaks detected.")
                return [], []

            # print(f"[DEBUG] Detected peaks: {peak_times}")
            # print(f"[DEBUG] Detected peak values: {peak_values}")

            return peak_times, peak_values
        except Exception as e:
            print(f"[PPGProcessor] Peak detection error: {e}")
            return [], []

    # def compute_hrv(self):
    #     if len(self.r) < 3:
    #         return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}
    #
    #     rr_intervals = np.diff(np.array(self.r)) * 1000  # w ms
    #     if len(rr_intervals) < 2:
    #         return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}
    #
    #     diff_rr = np.diff(rr_intervals)
    #     # print(f"[DEBUG] RR intervals (ms): {rr_intervals}")
    #     # print(f"[DEBUG] Differences between RR intervals: {diff_rr}")
    #     rmssd = np.sqrt(np.mean(diff_rr ** 2))
    #     sdnn = np.std(rr_intervals)
    #     nn50 = np.sum(np.abs(diff_rr) > 50)
    #     pnn50 = 100.0 * nn50 / len(diff_rr)
    #
    #     # print(f"[DEBUG] HRV Metrics: RMSSD={rmssd}, SDNN={sdnn}, pNN50={pnn50}")
    #     return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50}

    def compute_hrv(self):
        if len(self.r) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

        rr_intervals = np.diff(np.array(self.r)) * 1000  # Convert to milliseconds
        if len(rr_intervals) < 2:
            return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        sdnn = np.std(rr_intervals)
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = 100.0 * nn50 / len(rr_intervals)

        return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50}

    def _normalize_window(self, window):
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val - min_val == 0:
            return np.zeros_like(window)
        return 2 * (window - min_val) / (max_val - min_val) - 1

    def reset(self):
        self.sample_buffer.clear()
        self.time_buffer.clear()