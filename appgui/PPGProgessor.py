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

        self.sample_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)

        #self.model = get_or_train_model()

        self.r = []

    def add_sample(self, sample, time):
        """
        Add new PPG sample and time.
        When buffer is full, return a PPGResult.
        Otherwise, return None.
        """
        self.sample_buffer.append(sample)
        self.time_buffer.append(time)

        if len(self.sample_buffer) == self.window_size:


            window_data = np.array(self.sample_buffer)
            time_data = np.array(self.time_buffer)
            filtered = self.process_func(window_data)

            peak_times, peak_values = self.detect_peaks(filtered, time_data)


            for i in peak_times:
                self.r.append(i)
            print(peak_times)

            diff = 60/np.mean(np.diff(np.array(self.r)))
            print("BMP: ", diff*1000)
            self.sample_buffer.clear()
            self.time_buffer.clear()

           # out_model = predict_ppg_segment(self.model, filtered)
            #print(out_model)

            return PPGResult(
                time_array=time_data,
                filtered_signal=filtered,
                raw_signal=window_data,
                peak_times=peak_times,
                peak_values=peak_values
            )
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
        Detect peaks using scipy.signal.find_peaks.
        Returns peak times and values.
        """
        try:

            peaks = nk.ppg_findpeaks(signal, sampling_rate=30)['PPG_Peaks']
            peak_times = time_array[peaks]
            peak_values = signal[peaks]
            return peak_times.tolist(), peak_values.tolist()
        except Exception as e:
            print(f"[PPGProcessor] Peak detection error: {e}")
            return [], []

    def _normalize_window(self, window):
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val - min_val == 0:
            return np.zeros_like(window)
        return 2 * (window - min_val) / (max_val - min_val) - 1

    def reset(self):
        self.sample_buffer.clear()
        self.time_buffer.clear()
