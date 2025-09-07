import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# prosta funkcja detekcji lokalnych maksimów (sąsiedztwo 1)
def local_maxima(x):
    # zwraca indeksy, gdzie x[i] > x[i-1] and x[i] > x[i+1]
    ix = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    return ix

# Wczytanie
dfEcg = pd.read_csv('ecg_data_aligned.csv')
dfPpg = pd.read_csv('ppg_data_aligned.csv')

# konwersje
dfEcg['ecg'] = pd.to_numeric(dfEcg['ecg'], errors='coerce').astype(int)
dfPpg['ppg'] = pd.to_numeric(dfPpg['ppg'], errors='coerce').astype(int)
dfEcg['time'] = pd.to_numeric(dfEcg['time'], errors='coerce')
dfPpg['time'] = pd.to_numeric(dfPpg['time'], errors='coerce')

# ucięcie pierwszych 8s (ms)
t0 = max(dfEcg['time'].iloc[0], dfPpg['time'].iloc[0])
cut_time = t0 + 8000
dfEcg = dfEcg[dfEcg['time'] >= cut_time].copy()
dfPpg = dfPpg[dfPpg['time'] >= cut_time].copy()

# względny czas w sekundach
dfEcg['t_s'] = (dfEcg['time'] - cut_time) / 1000.0
dfPpg['t_s'] = (dfPpg['time'] - cut_time) / 1000.0

# Interpolacja PPG na siatkę czasową ECG (linearna)
ecg_times = dfEcg['t_s'].values
if len(dfPpg) < 2:
    raise RuntimeError("Za mało próbek PPG do interpolacji.")
ppg_interp = np.interp(ecg_times, dfPpg['t_s'].values, dfPpg['ppg'].values)

# Zrób okno referencyjne (8-12s po ucięciu) — uwaga: to teraz w sekundach
ref_start, ref_end = 8.0, 12.0
mask_ref = (ecg_times >= ref_start) & (ecg_times <= ref_end)
if mask_ref.sum() == 0:
    raise RuntimeError("Brak próbek ECG w oknie referencyjnym — poszerz okno lub sprawdź dane.")

ecg_window = dfEcg['ecg'].values[mask_ref]
ppg_window_interp = ppg_interp[mask_ref]

# baseline = mediana w oknie
ecg_baseline = np.median(ecg_window)
ppg_baseline = np.median(ppg_window_interp)

# znajdź lokalne maksima w oknie (na siatce ecg_times)
local_ix = local_maxima(ecg_window)
local_ix = local_ix[(ecg_times[mask_ref][local_ix] >= 0)]  # filtr bezpieczeństwa
# jeśli lokal_ix puste -> użyj top-percentyl
if len(local_ix) >= 3:
    ecg_peaks = ecg_window[local_ix]
    ppg_peaks = ppg_window_interp[local_ix]
else:
    # top 5% wartości jako "pikowe"
    k = max(1, int(0.05 * len(ecg_window)))
    ecg_peaks = np.sort(ecg_window)[-k:]
    ppg_peaks = np.sort(ppg_window_interp)[-k:]

# peak_mean = średnia wykrytych szczytów
ecg_peak_mean = np.mean(ecg_peaks)
ppg_peak_mean = np.mean(ppg_peaks)

amp_ecg = ecg_peak_mean - ecg_baseline
amp_ppg = ppg_peak_mean - ppg_baseline

# jeżeli amp_ppg bliskie 0 -> fallback na std ratio
if np.isclose(amp_ppg, 0) or np.isnan(amp_ppg):
    ecg_std = np.std(ecg_window)
    ppg_std = np.std(ppg_window_interp)
    scale = ecg_std / ppg_std if ppg_std and not np.isnan(ppg_std) else 1.0
else:
    scale = amp_ecg / amp_ppg

# zastosuj skalowanie (na całej interpolowanej PPG)
ppg_scaled_on_ecg = (ppg_interp - ppg_baseline) * scale + ecg_baseline

print(f"ecg_baseline={ecg_baseline:.2f}, ppg_baseline={ppg_baseline:.2f}")
print(f"ecg_peak_mean={ecg_peak_mean:.2f}, ppg_peak_mean={ppg_peak_mean:.2f}")
print(f"scale={scale:.4f}, amp_ecg={amp_ecg:.2f}, amp_ppg={amp_ppg:.2f}")

# Rysuj: ECG (raw), PPG (po interpolacji i skalowaniu). Zaznacz wykryte piki w oknie.
plt.figure(figsize=(12,5))
plt.plot(ecg_times, dfEcg['ecg'].values, label='ECG (raw)', linewidth=1)
plt.plot(ecg_times, ppg_scaled_on_ecg, label='PPG (interpolated + scaled)', linewidth=1)

# pokaz wykryte piki (na siatce użytej do liczeń)
# jeżeli korzystaliśmy z lokal_ix to one są indeksami względem okna — przemapuj na globalne indeksy

## PIKI

# if len(local_ix) >= 1:
#     global_peak_ix = np.where(mask_ref)[0][local_ix]
#     plt.scatter(ecg_times[global_peak_ix], dfEcg['ecg'].values[global_peak_ix],
#                 color='red', s=30, label='ECG detected peaks')
#     plt.scatter(ecg_times[global_peak_ix], ppg_scaled_on_ecg[global_peak_ix],
#                 color='orange', s=30, label='PPG at ECG peaks')

# zaznacz okno referencyjne
# plt.axvspan(ref_start, ref_end, color='gray', alpha=0.12)

plt.xlabel('Time [s after cut]')
plt.ylabel('Signal (units)')
plt.title('ECG vs PPG (PPG interpolated to ECG times, scaled to match peaks)')
plt.ylim(-500, 1750)
plt.legend()
plt.tight_layout()
plt.show()
