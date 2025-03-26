import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


def calculate_RSA_from_RR(csv_file_path, hf_range=(0.15, 0.4), fs=1.0):
    """
    Funkcja do obliczenia RSA (Respiratory Sinus Arrhythmia) z danych interwałów R-R zapisanych w pliku CSV.

    csv_file_path: str
        Ścieżka do pliku CSV zawierającego kolumnę 'RR_intervals'.

    hf_range: tuple (default=(0.15, 0.4))
        Zakres częstotliwości, który będzie użyty do obliczeń RSA (domyślnie 0.15-0.4 Hz).

    fs: float (default=1.0)
        Częstotliwość próbkowania, domyślnie 1.0 Hz (jedno próbkowanie na sekundę).

    Zwraca:
        hf_power: float
            Moc w paśmie HF (0.15-0.4 Hz), która odpowiada RSA.
    """

    # Wczytanie danych z pliku CSV
    df = pd.read_csv(csv_file_path)

    if 'R-R' not in df.columns:
        raise ValueError("Brak kolumny 'RR_intervals' w pliku CSV.")

    # Przygotowanie danych: konwertowanie na numpy array i usunięcie NaN
    RR_intervals_np = df['R-R'].dropna().to_numpy()

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

    # 3. Wykres analizy częstotliwościowej
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_fft_result)
    plt.title("Analiza częstotliwościowa (FFT) - Moc w funkcji częstotliwości")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

    # 4. Wyświetlenie mocy w paśmie HF związanej z RSA
    print(f"Moc w paśmie HF (0.15-0.4 Hz) związanej z RSA: {hf_power:.4f}")

    return hf_power


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


def calculate_breathing_rate_from_RR(RR_intervals_np, hf_range=(0.15, 0.4), fs=360):
    """
    Funkcja do obliczenia częstotliwości oddechowej z danych interwałów R-R zapisanych w pliku CSV.

    csv_file_path: str
        Ścieżka do pliku CSV zawierającego kolumnę 'RR_intervals'.

    hf_range: tuple (default=(0.15, 0.4))
        Zakres częstotliwości, który będzie użyty do obliczeń RSA (domyślnie 0.15-0.4 Hz).

    fs: float (default=1.0)
        Częstotliwość próbkowania, domyślnie 1.0 Hz (jedno próbkowanie na sekundę).

    Zwraca:
        breathing_rate: float
            Częstotliwość oddechowa w oddechach na minutę.
    """

    # Wczytanie danych z pliku CSV

    # 1. Analiza częstotliwościowa (FFT)
    N = len(RR_intervals_np)
    fft_result = fft(RR_intervals_np)

    # Obliczanie częstotliwości
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # Tylko dodatnie częstotliwości
    positive_freqs = freqs[:N // 2]
    positive_fft_result = np.abs(fft_result[:N // 2])

    # 2. Wyszukiwanie dominującej częstotliwości w paśmie oddechowym (0.1–0.4 Hz)
    breath_freqs = positive_freqs[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])]
    breath_amplitudes = positive_fft_result[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])]

    # Dominująca częstotliwość oddechowa
    max_amplitude_index = np.argmax(breath_amplitudes)
    breathing_freq = breath_freqs[max_amplitude_index]

    # Częstotliwość oddechowa w oddechach na minutę (bpm - breaths per minute)
    breathing_rate = breathing_freq * 60  # konwersja z Hz na oddechy na minutę

    # 3. Wykres analizy częstotliwościowej
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_fft_result)
    plt.title("Analiza częstotliwościowa (FFT) - Moc w funkcji częstotliwości")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

    # 4. Wyświetlenie częstotliwości oddechowej
    print(f"Częstotliwość oddechowa: {breathing_rate:.2f} oddechów na minutę")

    return breathing_rate