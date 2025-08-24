from collections import deque
import numpy as np
import torch
import neurokit2 as nk
from r_neural import get_model, predict


class ECGOutput:
    def __init__(self, x_peaks, y_peaks, ecg_filtered, peak_unix_times, hrv=None):
        self.x_peaks = x_peaks
        self.y_peaks = y_peaks
        self.ecg_filtered = ecg_filtered
        self.peak_unix_times = peak_unix_times
        self.hrv = hrv or {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}


class ECGProcessor:
    def __init__(self, window_size=100, hrv_window_sec=60, sampling_rate=130):
        """
        window_size: liczba próbek na okno przetwarzania sygnału
        hrv_window_sec: długość przesuwającego się okna HRV (w sekundach)
        """
        self.window_size = window_size
        self.sample_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        self.r_peak_times = deque()  # dynamiczne okno
        self.hrv_window_sec = hrv_window_sec
        self.sampling_rate = sampling_rate

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = torch.device(device)

        self.model = get_model(self.device)

        self.timestamps = []
        self.unix_time_buffer = deque(maxlen=window_size)

    def add_sample(self, sample, timestamp, time):
        self.sample_buffer.append(sample)
        self.time_buffer.append(time)
        self.unix_time_buffer.append(timestamp)

        if len(self.sample_buffer) == self.window_size:
            window_data = np.array(self.sample_buffer)
            time_data = np.array(self.time_buffer)
            unix_data = np.array(self.unix_time_buffer)
            result = self.process_func(window_data, time_data, unix_data)
            self.sample_buffer.clear()
            self.time_buffer.clear()
            self.unix_time_buffer.clear()
            return result
        return None

    def process_func(self, window_data, time_data, unix_data):
        peaks = predict(self.device, self.model, window_data.astype(np.float32))

        peaks_x = []
        peaks_y = []
        peak_unix_times = []
        for i, is_peak in enumerate(peaks):
            if is_peak == 1:
                peak_time = time_data[i]
                peaks_x.append(peak_time)
                peaks_y.append(window_data[i])
                self.r_peak_times.append(peak_time)
                peak_unix_times.append(unix_data[i])

        # --- Przesuwające się okno HRV ---
        self._trim_r_peaks()

        ecg_filtered = nk.signal_filter(
            window_data,
            sampling_rate=self.sampling_rate,
            lowcut=0.5,
            highcut=45,
            method="butterworth",
            order=5
        )

        hrv = self.compute_hrv()
        return ECGOutput(peaks_x, peaks_y, self._normalize_window(ecg_filtered), peak_unix_times, hrv)

    def _trim_r_peaks(self):
        """Usuwa stare R-peaki spoza okna HRV."""
        if not self.r_peak_times:
            return
        current_time = self.r_peak_times[-1]
        while self.r_peak_times and (current_time - self.r_peak_times[0] > self.hrv_window_sec):
            self.r_peak_times.popleft()

    def compute_hrv(self):
        if len(self.r_peak_times) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

        rr_intervals = np.diff(np.array(self.r_peak_times)) * 1000  # w ms
        if len(rr_intervals) < 2:
            return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        sdnn = np.std(rr_intervals)
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = 100.0 * nn50 / len(diff_rr)

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
        self.r_peak_times.clear()
        self.unix_time_buffer.clear()
