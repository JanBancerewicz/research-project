from collections import deque
import numpy as np
import torch
import neurokit2 as nk
from r_neural import get_model, predict

class ECGOutput:
    def __init__(self, x_peaks, y_peaks, ecg_filtered):
        self.x_peaks = x_peaks
        self.y_peaks = y_peaks
        self.ecg_filtered = ecg_filtered


class ECGProcessor:
    def __init__(self, window_size=100):
        """
        window_size: number of samples per window
        process_func: function to process each window (e.g. normalize)
        """
        self.window_size = window_size
        self.sample_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = torch.device(device)

        self.model = get_model(self.device)

    def add_sample(self, sample, time):
        """
        Add new ECG sample and its time value.
        When the buffer fills, process the window and return (time_values, processed_data).
        Otherwise return None.
        """
        self.sample_buffer.append(sample)
        self.time_buffer.append(time)

        if len(self.sample_buffer) == self.window_size:
            window_data = np.array(self.sample_buffer)
            time_data = np.array(self.time_buffer)
            processed = self.process_func(window_data, time_data)
            self.sample_buffer.clear()
            self.time_buffer.clear()
            return processed
        else:
            return None

    def process_func(self, window_data, time_data):
        peaks = predict(self.device, self.model, np.array(window_data, dtype=np.float32))

        peaks_x = []
        peaks_y = []
        for i in range(len(peaks)):
            if peaks[i]  == 1:
                peaks_x.append(time_data[i])
                peaks_y.append(window_data[i])

        ecg_filtered = nk.signal_filter(window_data, sampling_rate=130, lowcut=0.5, highcut=45, method="butterworth", order=5)
        return ECGOutput(peaks_x, peaks_y, self._normalize_window(ecg_filtered))

    def _normalize_window(self, window):
        """
        Normalize window data to [-1, 1]
        """
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val - min_val == 0:
            return np.zeros_like(window)
        return 2 * (window - min_val) / (max_val - min_val) - 1

    def reset(self):
        """
        Clear buffered data.
        """
        self.sample_buffer.clear()
        self.time_buffer.clear()
