from collections import deque
from dataclasses import dataclass
import numpy as np
import os  # <--- Import os
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
        """
        self.polyorder = polyorder
        self.window_size = window_size
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.time_unix = []

        self.sample_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)

        # --- PATH FIX: Ustalanie ścieżek absolutnych ---
        # 1. Gdzie jest ten plik (appgui/PPGProcessor.py)?
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. Folder główny projektu (research-project)
        project_root = os.path.dirname(current_script_dir)

        # 3. Absolutna ścieżka do modelu (w głównym folderze)
        model_abs_path = os.path.join(project_root, "ppg_peak_model.pth")

        # 4. Absolutna ścieżka do danych treningowych (cnn/ppg/train_data)
        # To jest potrzebne tylko jeśli model nie istnieje i trzeba trenować
        data_dir_abs = os.path.join(project_root, "cnn", "ppg", "train_data")

        print(f"DEBUG: PPGProcessor szuka modelu w: {model_abs_path}")

        # Parametry
        SEGMENT_LENGTH = window_size
        MAX_SEGMENTS = 5000  # Zmniejszyłem z 10000 dla szybszego startu (jeśli będzie musiał trenować)
        EPOCHS = 20  # Zmniejszyłem z 200 na 10! (To był powód "mielenia")
        BATCH_SIZE = 32
        LR = 0.001
        MAX_FILES = None

        self.model = get_or_train_model(
            model_path=model_abs_path,
            data_dir=data_dir_abs,
            segment_length=SEGMENT_LENGTH,
            max_segments=MAX_SEGMENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_files=MAX_FILES
        )

        self.r = []
        self.r_for_rr = []

    def add_sample(self, sample, time, time_unix):
        self.sample_buffer.append(sample)
        self.time_buffer.append(time)
        self.time_unix.append(time_unix)

        if len(self.sample_buffer) == self.window_size:
            window_data = np.array(self.sample_buffer)
            time_data = np.array(self.time_buffer)
            unix_data = np.array(self.time_unix)

            filtered = self.process_func(window_data)
            peak_times, peak_values = self.detect_peaks(filtered, time_data)

            peak_unix_times = []
            for pt in peak_times:
                idx = np.where(time_data == pt)[0]
                if len(idx) > 0:
                    peak_unix_times.append(unix_data[idx[0]])
                else:
                    peak_unix_times.append(None)

            for peak_time in peak_times:
                self.r.append(peak_time)
                self.r_for_rr.append(peak_time)

            hrv = self.compute_hrv()
            # print(f"HRV Metrics: {hrv}") # Zakomentowane, żeby nie śmiecić w konsoli

            self.sample_buffer.clear()
            self.time_buffer.clear()
            self.time_unix.clear()

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
        return self._normalize_window(self.bandpass_filter(window_data))

    def detect_peaks(self, signal, time_array):
        try:
            out = predict_ppg_segment(self.model, signal)
            peak_times = []
            peak_values = []
            for i in range(len(out)):
                if out[i]:
                    # print("------------", time_array[i], signal[i]) # Opcjonalnie zakomentuj
                    peak_times.append(time_array[i])
                    peak_values.append(signal[i])

            if len(peak_times) == 0:
                # print("[PPGProcessor] No peaks detected.")
                return [], []

            return peak_times, peak_values
        except Exception as e:
            print(f"[PPGProcessor] Peak detection error: {e}")
            return [], []

    def compute_hrv(self):
        if len(self.r_for_rr) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "rr_intervals": []}

        rr_intervals = np.diff(np.array(self.r_for_rr))
        if len(rr_intervals) < 2:
            return {"rmssd": 0.0, "sdnn": 0.0, "rr_intervals": []}

        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        sdnn = np.std(rr_intervals)

        self.r_for_rr = [self.r_for_rr[-1]]

        return {"rmssd": rmssd, "sdnn": sdnn, "rr_intervals": rr_intervals}

    def _normalize_window(self, window):
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val - min_val == 0:
            return np.zeros_like(window)
        return 2 * (window - min_val) / (max_val - min_val) - 1

    def reset(self):
        self.sample_buffer.clear()
        self.time_buffer.clear()
        self.r.clear()