import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft



def calculate_RSA_from_RR(RR_intervals_np, hf_range=(0.15, 0.4), breath_rate_range=(0.08, 0.12), fs=130.0):
    """
    Funkcja do obliczenia RSA (Respiratory Sinus Arrhythmia) z danych interwałów R-R zapisanych w pliku CSV.

    RR_intervals_np: np.array
        NumPy array zawierający interwały R-R.

    hf_range: tuple (default=(0.15, 0.4))
        Zakres częstotliwości, który będzie użyty do obliczeń RSA (domyślnie 0.15-0.4 Hz).

    breath_rate_range: tuple (default=(0.08, 0.12))
        Zakres częstotliwości, który będzie użyty do obliczeń częstotliwości oddechu (domyślnie 0.08-0.12 Hz).

    fs: float (default=130.0)
        Częstotliwość próbkowania, domyślnie 130 Hz.

    Zwraca:
        hf_power: float
            Moc w paśmie HF (0.15-0.4 Hz), która odpowiada RSA.

        breath_rate_power: float
            Moc w paśmie oddechowym (0.08-0.12 Hz).
    """
    # Przygotowanie danych: konwertowanie na numpy array i usunięcie NaN
    RR_intervals_np = np.asarray(RR_intervals_np)
    RR_intervals_np = RR_intervals_np[~np.isnan(RR_intervals_np)]

    # 1. Analiza częstotliwościowa (FFT)
    N = len(RR_intervals_np)
    fft_result = fft(RR_intervals_np)

    # Obliczanie częstotliwości
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # Tylko dodatnie częstotliwości
    positive_freqs = freqs[:N // 2]
    positive_fft_result = np.abs(fft_result[:N // 2])

    # 2. Wyszukiwanie mocy w paśmie HF (0.15-0.4 Hz)
    hf_freqs = positive_freqs[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])]
    hf_power = np.sum(positive_fft_result[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])])

    # 3. Wyszukiwanie mocy w paśmie oddechowym (0.08-0.12 Hz)
    breath_rate_freqs = positive_freqs[(positive_freqs >= breath_rate_range[0]) & (positive_freqs <= breath_rate_range[1])]
    breath_rate_power = positive_fft_result[ (positive_freqs >= breath_rate_range[0]) & (positive_freqs <= breath_rate_range[1])]

    # Maksymalna moc w paśmie oddechowym
    max_breath_power_index = np.argmax(breath_rate_power)
    breathing_freq = breath_rate_freqs[max_breath_power_index]
    print(f"Moc w paśmie HF (0.15-0.4 Hz) związanej z RSA: {hf_power:.4f}")
    print(f"Moc w paśmie oddechowym (0.08-0.12 Hz): {breathing_freq:.8f}")

    return hf_power, hf_power

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


import numpy as np
from scipy.fft import fft

def calculate_breathing_rate_from_RR(rr_intervals, breath_rate_range=(0.08, 0.12), fs=130):
    sampling_frequency = 130.0  # 1 sample per second

    # Perform Fast Fourier Transform (FFT) to identify the dominant frequency (breathing rate)
    n = len(rr_intervals)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_frequency)  # Get frequency bins
    fft_values = np.fft.fft(rr_intervals)  # FFT of RR intervals

    # Only keep positive frequencies
    positive_frequencies = frequencies[:n // 2]
    positive_fft_values = np.abs(fft_values[:n // 2])

    # Find the respiratory frequency band (0.1-0.5 Hz)
    respiratory_band = (positive_frequencies >= 0.1) & (positive_frequencies <= 0.5)

    # Check if the respiratory band has any valid frequencies
    if respiratory_band.any():
        # Find the frequency with the maximum amplitude in the respiratory band
        breathing_frequency = positive_frequencies[respiratory_band][np.argmax(positive_fft_values[respiratory_band])]

        # Convert frequency to breaths per minute
        breathing_rate = breathing_frequency * 60  # in breaths per minute
        print(f"Estimated Breathing Rate: {breathing_rate:.2f} breaths per minute")
        return breathing_rate