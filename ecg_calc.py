import numpy as np
from collections import deque

from scipy.interpolate import interp1d
from scipy.signal import welch


class ECGProcessor:
    def __init__(self, rr_buffer_size=5):
        self.rr_intervals = []

    def add_sample(self, rr):
        """Dodaj nowy odstęp RR (w milisekundach) i oblicz cechy HRV."""
        for i in rr:
            if 300 < i <= 1000:
                self.rr_intervals.append(i)
        if len(self.rr_intervals) > 30:
            self.rr_intervals = self.rr_intervals[len(self.rr_intervals)-30:]

        if len(self.rr_intervals) < 3:
            return {}

        rr = np.array(self.rr_intervals)
        rsa, hf = self.compute_rsa_from_rr(rr)
        return {
            'rmssd': self.compute_rmssd(rr),
            'sdnn': self.compute_sdnn(rr),
            'rsa': rsa,
            'hf': hf,
            'hr': 60000.0 / np.mean(rr) if np.mean(rr) > 0 else 0.0
        }

    @staticmethod
    def compute_rmssd(rr):
        diff_rr = np.diff(rr)
        squared_diff = diff_rr ** 2
        return np.sqrt(np.mean(squared_diff)) if len(squared_diff) > 0 else 0.0

    @staticmethod
    def compute_sdnn(rr):
        return float(np.std(rr)) if len(rr) > 1 else 0.0

    @staticmethod
    def compute_rsa(rr):
        return float(np.std(np.diff(rr))) if len(rr) > 2 else 0.0

    def compute_hf_power(self, rr_signal, fs=4.0, hf_range=(0.15, 0.4)):
        """Oblicza moc widmową w paśmie HF."""
        freqs, psd = welch(rr_signal, fs=fs, nperseg=256)
        hf_band = (freqs >= hf_range[0]) & (freqs <= hf_range[1])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band]) * 1e6  # z s² na ms²
        return hf_power

    def detrend_signal(self, signal):
        """Usuwa średnią z sygnału."""
        return signal - np.mean(signal)

    def get_time_series_from_rr(self, rr_intervals, fs=4.0):
        """Interpoluje sygnał RR do równomiernej siatki czasowej."""
        rr_s = np.array(rr_intervals) / 1000.0
        time = np.cumsum(rr_s)
        time -= time[0]  # start od 0

        interpolated_time = np.arange(0, time[-1], 1.0 / fs)
        interpolator = interp1d(time, rr_s, kind='cubic', fill_value="extrapolate")
        rr_interp = interpolator(interpolated_time)
        return interpolated_time, rr_interp



    def compute_rsa_from_rr(self, rr_intervals, fs=4.0):
        """Główna funkcja: oblicza RSA (ln HF) z RR interwałów."""
        try:
            time, rr_interp = self.get_time_series_from_rr(rr_intervals, fs)
            rr_detrended = self.detrend_signal(rr_interp)
            hf_power = self.compute_hf_power(rr_detrended, fs)

            rsa_ln = np.log(hf_power) if hf_power > 0 else float('-inf')
            return rsa_ln, hf_power
        except:
            return 0

