import numpy as np
from collections import deque

from scipy.interpolate import interp1d
from scipy.signal import welch


class ECGProcessor:
    def __init__(self, rr_buffer_size=5):
        self.rr_intervals = []
        self.r_ampl = []
        self.hr_history = []

    def add_sample(self, rr, r_amp):
        """Dodaj nowy odstęp RR (w milisekundach) i oblicz cechy HRV."""
        for idx, i in enumerate(rr):
            if 300 < i <= 1000:
                self.rr_intervals.append(i)
                self.r_ampl.append(r_amp[idx])
        if len(self.rr_intervals) > 20:
            self.rr_intervals = self.rr_intervals[5:]
            self.r_ampl = self.r_ampl[5:]

        if len(self.rr_intervals) < 3:
            return {}

        rr = np.array(self.rr_intervals)
        if len(self.hr_history) > 1:
            h = 60000.0 / np.mean(rr) if np.mean(rr) > 0 else 0.0
            if self.hr_history[-1] != h:
                self.hr_history.append(h)

        hr_slope = self.hr_history[-1] - self.hr_history[-2] if len(self.hr_history) >= 2 else 0.0

        return {
            'rmssd': self.compute_rmssd(rr),
            'sdnn': self.compute_sdnn(rr),
            'edr_mean': self.compute_edr_mean(self.r_ampl),
            'hr': 60000.0 / np.mean(rr) if np.mean(rr) > 0 else 0.0,
            'rr_slope': np.diff(rr)[-1],
        }

    def compute_lf_hf(self, rr_intervals, fs=4.0):
        """
        Oblicza moce LF i HF oraz stosunek LF/HF na podstawie RR (w ms).
        fs = częstość próbkowania interpolowanego sygnału RR (np. 4 Hz)
        """
        rr = np.array(rr_intervals)
        if len(rr) < 4:
            return 0.0, 0.0, 0.0  # brak danych

        # interpolacja RR na równomierną siatkę czasu
        time_rr = np.cumsum(rr) / 1000.0  # czas w sekundach
        interpolated_time = np.arange(time_rr[0], time_rr[-1], 1.0 / fs)
        interpolated_rr = np.interp(interpolated_time, time_rr, rr)

        f, psd = welch(interpolated_rr, fs=fs, nperseg=min(256, len(interpolated_rr)))

        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        lf_power = np.trapz(psd[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
        hf_power = np.trapz(psd[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])

        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

        return lf_power, hf_power, lf_hf_ratio
    @staticmethod
    def compute_edr_mean(edr_signal):
        edr = np.array(edr_signal)
        return float(np.mean(edr)) if len(edr) > 0 else 0.0
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
            return 0, 0

