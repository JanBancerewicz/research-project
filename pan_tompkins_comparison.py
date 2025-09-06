import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt
import os

CSV_PATH = "data/ecg/ECG1.csv"

# --- load + numeric coercion (jak wcześniej) ---
df = pd.read_csv(CSV_PATH, header=None, names=["time", "ecg"])
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["ecg"]  = pd.to_numeric(df["ecg"], errors="coerce")
df = df.dropna().reset_index(drop=True)
time = df["time"].values
ecg  = df["ecg"].values

# --- time units -> seconds, fs ---
dt_raw = np.median(np.diff(time))
if dt_raw > 0.2:          # heurystyka: jeżeli wartości duże, to ms
    time_sec = time / 1000.0
else:
    time_sec = time.copy()
dt = np.median(np.diff(time_sec))
fs = 1.0 / dt
print(f"dt = {dt:.6f} s -> fs = {fs:.2f} Hz")

# --- zero-phase bandpass (filtfilt) to avoid phase shift ---
def bandpass_filtfilt(signal, fs, lowcut=5.0, highcut=35.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# tune lowcut/highcut jeśli potrzebujesz (np. 0.5-40 dla slow waves)
ecg_bp = bandpass_filtfilt(ecg, fs, lowcut=5.0, highcut=35.0, order=3)

# --- optional extra cleaning (neurokit) ---
ecg_cleaned = nk.ecg_clean(ecg_bp, sampling_rate=fs, method="neurokit")

# --- detect with neurokit (Pan-Tompkins) ---
signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="pantompkins1985")

# robust extraction of indices:
def extract_rpeaks(signals, info):
    r = None
    if "ECG_R_Peaks" in info:
        r = info["ECG_R_Peaks"]
    elif "ECG_R_Peaks" in signals:
        mask = np.asarray(signals["ECG_R_Peaks"])
        if mask.dtype == bool or set(np.unique(mask)).issubset({0,1}):
            r = np.where(mask)[0]
        else:
            r = np.asarray(mask)
    else:
        r = np.array([], dtype=int)
    return np.asarray(r, dtype=int)

rpeaks_nn = extract_rpeaks(signals, info)
print(f"Neurokit zwrócił {len(rpeaks_nn)} pików.")

# --- refinement: przesuwamy wykryte piki do najbliższego lokalnego maksimum (abs) ---
# def refine_peaks(rpeaks, signal, fs, window_ms=40):
#     window = max(1, int(window_ms/1000.0 * fs))
#     refined = []
#     for r in rpeaks:
#         lo = max(0, r - window)
#         hi = min(len(signal) - 1, r + window)
#         local = signal[lo:hi+1]
#         if local.size == 0:
#             continue
#         # znajdź indeks największej amplitudy bezwzględnej (obsługuje odwrócone piki)
#         local_idx = int(np.argmax(np.abs(local)))
#         refined.append(lo + local_idx)
#     # unikaj duplikatów i posortuj
#     refined = np.unique(np.array(refined, dtype=int))
#     return refined
#
# rpeaks_refined = refine_peaks(rpeaks_nn, ecg_cleaned, fs, window_ms=50)

def refine_peaks_polarity(rpeaks, signal, fs, window_ms=50):
    """
    Dla każdego piku:
    - sprawdź średni znak sygnału w małym oknie wokół wykrycia (polarity)
    - jeśli polarity > 0: wybierz lokalne maksimum w oknie
      jeśli polarity < 0: wybierz lokalne minimum w oknie
    - zwróć posortowaną, unikalną listę indeksów
    """
    window = max(1, int(window_ms/1000.0 * fs))
    refined = []
    for r in rpeaks:
        lo = max(0, r - window)
        hi = min(len(signal) - 1, r + window)
        local = signal[lo:hi+1]
        if local.size == 0:
            continue
        # określ polaryzację przez średnią wartości blisko wykrycia (np. +/- 3 próbki)
        small = signal[max(0, r-3): min(len(signal), r+4)]
        polarity = np.nanmean(small)
        if polarity >= 0:
            # szukamy maksimum (pozycja najwyższej wartości)
            local_idx = int(np.argmax(local))
        else:
            # szukamy minimum (najniższej wartości)
            local_idx = int(np.argmin(local))
        refined.append(lo + local_idx)

    refined = np.unique(np.array(refined, dtype=int))
    return refined

rpeaks_refined = refine_peaks_polarity(rpeaks_nn, ecg_cleaned, fs, window_ms=50)



# --- porównanie przesunięcia (w ms) ---
if len(rpeaks_nn) == len(rpeaks_refined) and len(rpeaks_nn) > 0:
    shifts_ms = (rpeaks_refined - rpeaks_nn) / fs * 1000.0
    print(f"Średnie przesunięcie po refinement: {np.mean(shifts_ms):+.1f} ms, mediana: {np.median(shifts_ms):+.1f} ms")
else:
    # jeśli ilości się różnią, policz różnicę par najbliższych (opcjonalnie)
    print("Liczba pików przed i po refinement różni się — sprawdź wyniki manualnie.")

# --- wykres: neurokit vs refined ---
plt.figure(figsize=(12,7))
ax1 = plt.subplot(2,1,1)
ax1.plot(time_sec, ecg, label="ECG (raw)", alpha=0.4)
ax1.plot(time_sec, ecg_cleaned, label="ECG (bandpassed + cleaned)")
if len(rpeaks_nn)>0:
    ax1.scatter(time_sec[rpeaks_nn], ecg_cleaned[rpeaks_nn], c="red", s=40, label="R (neurokit)")
if len(rpeaks_refined)>0:
    ax1.scatter(time_sec[rpeaks_refined], ecg_cleaned[rpeaks_refined], c="lime", s=40, marker="x", label="R (refined)")
ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Amplitude")
ax1.legend(); ax1.grid(True)
ax1.set_title("Detekcja: czerwone = neurokit, zielone = po refinement")

# zoom pierwszych kilku sekund
zoom_s = 6
mask = time_sec <= (time_sec[0] + zoom_s)
ax2 = plt.subplot(2,1,2)
ax2.plot(time_sec[mask], ecg_cleaned[mask], label="ECG (cleaned)")
if len(rpeaks_nn)>0:
    sel = rpeaks_nn[(time_sec[rpeaks_nn] <= time_sec[0] + zoom_s)]
    ax2.scatter(time_sec[sel], ecg_cleaned[sel], c="red", s=40)
if len(rpeaks_refined)>0:
    sel2 = rpeaks_refined[(time_sec[rpeaks_refined] <= time_sec[0] + zoom_s)]
    ax2.scatter(time_sec[sel2], ecg_cleaned[sel2], c="lime", s=60, marker="x")
ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Amplitude"); ax2.grid(True)
plt.tight_layout()
plt.savefig("ecg_refined_comparison.png")
print("Zapisano wykres jako ecg_refined_comparison.png")
plt.show()
