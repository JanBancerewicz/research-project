import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
import neurokit2 as nk

# === 1. Wczytanie danych ===
data = pd.read_csv('ppg_data.csv')
ppg = np.array(data['ppg'])
time = np.array(data['time'])

# === 2. Przygotowanie sygnału ===
ppg = ppg[20:]
time = time[20:]
time -= time[0]

# === 3. Parametry ===
fs = 30  # częstotliwość próbkowania w Hz

# === 4. Definicje filtrów ===
def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal_data)

def lowpass_filter(signal_data, fs, cutoff=5.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal_data)

def median_filter(signal_data, kernel_size=5):
    return medfilt(signal_data, kernel_size=kernel_size)

# === 5. Zastosowanie filtrów ===
filtered_savgol = savgol_filter(ppg, window_length=31, polyorder=3)
filtered_bandpass = bandpass_filter(ppg, fs)
filtered_lowpass = lowpass_filter(ppg, fs)
filtered_median = median_filter(ppg, kernel_size=5)

# === 6. Detekcja pików ===
peaks_savgol = nk.ppg_findpeaks(filtered_savgol, sampling_rate=fs)['PPG_Peaks']
peaks_bandpass = nk.ppg_findpeaks(filtered_bandpass, sampling_rate=fs)['PPG_Peaks']
peaks_lowpass = nk.ppg_findpeaks(filtered_lowpass, sampling_rate=fs)['PPG_Peaks']
peaks_median = nk.ppg_findpeaks(filtered_median, sampling_rate=fs)['PPG_Peaks']

# === 8. Funkcja do obliczania BPM i HRV, gdy czas jest w milisekundach ===
def compute_bpm_hrv(peaks_indices, time_vector_ms):
    time_vector_s = time_vector_ms / 1000  # konwersja do sekund
    peak_times = time_vector_s[peaks_indices]
    rr_intervals = np.diff(peak_times)  # w sekundach
    rr_ms = rr_intervals * 1000  # w milisekundach

    bpm = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan
    hrv_sdnn = np.std(rr_ms) if len(rr_ms) > 1 else np.nan

    return bpm, hrv_sdnn


# === 9. Obliczenia dla każdego filtru ===
bpm_sg, hrv_sg = compute_bpm_hrv(peaks_savgol, time)
bpm_bp, hrv_bp = compute_bpm_hrv(peaks_bandpass, time)
bpm_lp, hrv_lp = compute_bpm_hrv(peaks_lowpass, time)
bpm_med, hrv_med = compute_bpm_hrv(peaks_median, time)

# === 10. Wyświetlenie wyników ===
print("Tętno (BPM) i HRV (SDNN w ms):")
print(f"Savitzky-Golay:     BPM = {bpm_sg:.1f}, HRV = {hrv_sg:.1f} ms")
print(f"Bandpass Butter:    BPM = {bpm_bp:.1f}, HRV = {hrv_bp:.1f} ms")
print(f"Lowpass Butter:     BPM = {bpm_lp:.1f}, HRV = {hrv_lp:.1f} ms")
print(f"Median Filter:      BPM = {bpm_med:.1f}, HRV = {hrv_med:.1f} ms")
# === 11. Przygotowanie danych do zapisu ===
output_df = pd.DataFrame({
    'time_ms': time,  # czas w milisekundach
    'ppg_bandpass': filtered_bandpass,
    'peak': 0  # domyślnie 0
})

# Oznaczenie pików jako 1
output_df.loc[peaks_bandpass, 'peak'] = 1

# === 12. Zapis do pliku CSV ===
output_df.to_csv('ppg_bandpass_with_peaks.csv', index=False)
print("Zapisano plik: ppg_bandpass_with_peaks.csv")

# === 7. Wykres z pikami ===
plt.figure(figsize=(14, 8))
plt.plot(time, filtered_savgol, label='Savitzky-Golay', linewidth=2)
plt.plot(time[peaks_savgol], filtered_savgol[peaks_savgol], 'ro', label='Peaki SG')

plt.plot(time, filtered_bandpass, label='Butterworth bandpass', linewidth=2)
plt.plot(time[peaks_bandpass], filtered_bandpass[peaks_bandpass], 'go', label='Peaki BP')

plt.plot(time, filtered_lowpass, label='Butterworth lowpass', linewidth=2)
plt.plot(time[peaks_lowpass], filtered_lowpass[peaks_lowpass], 'mo', label='Peaki LP')

plt.plot(time, filtered_median, label='Medianowy', linewidth=2)
plt.plot(time[peaks_median], filtered_median[peaks_median], 'ko', label='Peaki Med')

plt.legend(loc='upper right')
plt.title("Detekcja pików w sygnale PPG po filtracji")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()
