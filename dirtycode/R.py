import asyncio
from bleak import BleakClient
from bleak import BleakScanner
import struct
import numpy as np
from scipy.fft import fft


# Funkcja do analizy częstotliwościowej z FFT i obliczania częstotliwości oddechowej
def calculate_breathing_rate_realtime(rr_data, fs=1.0, hf_range=(0.15, 0.4)):
    """
    Analizuje dane interwałów R-R w czasie rzeczywistym i oblicza częstotliwość oddechową.

    rr_data: list or np.array
        Lista interwałów R-R w sekundach (dane wejściowe z bufora).

    fs: float (default=1.0)
        Częstotliwość próbkowania, domyślnie 1 Hz.

    hf_range: tuple (default=(0.15, 0.4))
        Zakres częstotliwości, który będzie użyty do obliczeń RSA (domyślnie 0.15-0.4 Hz).

    Zwraca:
        breathing_rate: float
            Częstotliwość oddechowa w oddechach na minutę.
    """
    N = len(rr_data)
    if N == 0:
        return 0

    # Analiza FFT
    fft_result = fft(rr_data)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # Tylko dodatnie częstotliwości
    positive_freqs = freqs[:N // 2]
    positive_fft_result = np.abs(fft_result[:N // 2])

    # Wyszukiwanie dominującej częstotliwości w paśmie oddechowym (0.15–0.4 Hz)
    breath_freqs = positive_freqs[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])]
    breath_amplitudes = positive_fft_result[(positive_freqs >= hf_range[0]) & (positive_freqs <= hf_range[1])]

    if len(breath_amplitudes) == 0:
        return 0

    # Dominująca częstotliwość oddechowa
    max_amplitude_index = np.argmax(breath_amplitudes)
    breathing_freq = breath_freqs[max_amplitude_index]

    # Częstotliwość oddechowa w oddechach na minutę (signal - breaths per minute)
    breathing_rate = breathing_freq * 60  # konwersja z Hz na oddechy na minutę

    return breathing_rate


# Funkcja do połączenia z Polar H10 i odczytu danych
async def connect_to_polar_h10(mac_address):
    # Znalezienie urządzenia
    device = await BleakScanner.discover()

    # Znalezienie odpowiedniego urządzenia Polar H10
    target_device = None
    for d in device:
        if d.address == mac_address:
            target_device = d
            break

    if target_device is None:
        print(f"Nie znaleziono urządzenia z adresem MAC {mac_address}")
        return

    async with BleakClient(target_device.address) as client:
        print(f"Połączono z {mac_address}")

        # Odczytuj dane z powiadomień (RR Interval)
        rr_intervals = []  # Lista do przechowywania interwałów R-R

        def rr_notification_handler(sender: int, data: bytearray):
            """Obsługuje powiadomienia o interwałach R-R z Polar H10"""
            try:
                if len(data) == 3:  # Zwykle dane interwału R-R to 3 bajty
                    rr_interval = struct.unpack("<H", data[1:3])[0] / 1000  # Konwertowanie na sekundy
                    rr_intervals.append(rr_interval)
                    print(f"RR Interval: {rr_interval:.4f} s")  # Drukowanie interwału R-R

                    # Po zgromadzeniu co najmniej 30 próbek R-R, oblicz częstotliwość oddechową

            except Exception as e:
                print(f"Błąd podczas przetwarzania danych R-R: {e}")

        # Subskrybuj powiadomienia o interwałach R-R
        rr_interval_char = "00002a53-0000-1000-8000-00805f9b34fb"  # UUID dla interwałów R-R (Polar H10)
        await client.start_notify(rr_interval_char, rr_notification_handler)

        try:
            # Czekaj na dane w tle (symulacja czasu rzeczywistego)
            while True:
                await asyncio.sleep(1)  # Oczekiwanie na powiadomienia
        except KeyboardInterrupt:
            print("Zakończono połączenie.")

