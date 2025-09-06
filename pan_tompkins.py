# pan_tompkins_custom.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
# import neurokit2 as nk  # opcjonalnie do porównania, jeśli zainstalowane; nieobiektywne

CSV_PATH = "data/ecg/ECG2.csv"  # dostosuj ścieżkę

# --- helper: load + fs detection ---
def load_ecg(csv_path):
    df = pd.read_csv(csv_path, header=0)  # pierwszy wiersz = header (timestamp,ecg)
    # normalize column names
    cols = [c.lower() for c in df.columns]
    if "timestamp" in cols:
        df.columns = ["time" if c.lower()=="timestamp" else c for c in df.columns]
    df = df.rename(columns={df.columns[0]: "time", df.columns[1]: "ecg"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["ecg"]  = pd.to_numeric(df["ecg"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    time = df["time"].values
    ecg  = df["ecg"].values
    # detect units (s or ms)
    dt_raw = np.median(np.diff(time))
    if dt_raw > 0.2:  # heurystyka — jeśli kroki duże, to ms -> zamieniamy na s
        time = time / 1000.0
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    return time, ecg, fs

# --- bandpass (zero-phase) ---
def bandpass(signal, fs, lowcut=5.0, highcut=15.0, order=3):
    nyq = 0.5 * fs
    low = max(0.0001, lowcut / nyq)
    high = min(0.9999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Pan-Tompkins style processing (derivative, squaring, integration) ---
def pan_tompkins_transform(signal, fs, maw_len_ms=150):
    # 1) derivative filter (Pan-Tompkins approx)
    # kernel corresponds to: [-1 -2 0 2 1] (approx derivative)
    kernel = np.array([-1, -2, 0, 2, 1], dtype=float)
    # normalize to sampling period: original PT scales by (1/8T) but normalization isn't necessary for peak detection
    derivative = np.convolve(signal, kernel, mode='same')
    # 2) squaring
    squared = derivative ** 2
    # 3) moving average / moving window integrator
    win_samples = max(1, int(round(maw_len_ms/1000.0 * fs)))
    window = np.ones(win_samples) / win_samples
    integrated = np.convolve(squared, window, mode='same')
    return derivative, squared, integrated, win_samples

# --- detect candidate QRS on integrated signal ---
def detect_qrs_from_integrated(integrated, fs, distance_ms=200, height=None, prominence=None):
    distance_samples = max(1, int(round(distance_ms/1000.0 * fs)))
    # if not provided, pick threshold as a fraction of max or median:
    if height is None:
        # robust init: use median + k * mad or a fraction of max
        height = max( (np.median(integrated) + 2*np.std(integrated)), 0.2*np.max(integrated) )
    peaks, props = find_peaks(integrated, distance=distance_samples, height=height, prominence=prominence)
    return peaks, props

# --- refine: map approximate peaks -> R-peaks on original cleaned signal ; dont need that---
def refine_peaks_to_r(original_signal, approx_peaks, fs, search_window_ms=75):
    window = max(1, int(round(search_window_ms/1000.0 * fs)))
    refined = []
    for p in approx_peaks:
        lo = max(0, p - window)
        hi = min(len(original_signal)-1, p + window)
        local = original_signal[lo:hi+1]
        if local.size == 0:
            continue
        # decide polarity by local mean near p
        small_lo = max(0, p-3)
        small_hi = min(len(original_signal)-1, p+3)
        polarity = np.nanmean(original_signal[small_lo:small_hi+1])
        if polarity >= 0:
            idx = int(np.argmax(local))
        else:
            idx = int(np.argmin(local))
        refined.append(lo + idx)
    refined = np.unique(np.array(refined, dtype=int))
    # enforce minimal RR (refractory): remove peaks closer than 200 ms keeping larger amplitude
    min_dist = int(round(0.2 * fs))
    if refined.size > 1:
        keep = [refined[0]]
        for r in refined[1:]:
            if r - keep[-1] < min_dist:
                # keep whichever has larger absolute amplitude
                if abs(original_signal[r]) > abs(original_signal[keep[-1]]):
                    keep[-1] = r
            else:
                keep.append(r)
        refined = np.array(keep, dtype=int)
    return refined

# --- full pipeline wrapper ---
def run_pan_tompkins_pipeline(time, ecg, fs,
                              bp_low=5.0, bp_high=15.0,
                              maw_ms=150, detect_distance_ms=200,
                              search_window_ms=75):
    ecg_bp = bandpass(ecg, fs, lowcut=bp_low, highcut=bp_high, order=3)
    derivative, squared, integrated, win_samples = pan_tompkins_transform(ecg_bp, fs, maw_len_ms=maw_ms)
    approx_peaks, props = detect_qrs_from_integrated(integrated, fs, distance_ms=detect_distance_ms)
    ## opcjonalny refinement, ktorego nie chcemy, bo liczy sie oryginalny algorytm
    #rpeaks = refine_peaks_to_r(ecg_bp, approx_peaks, fs, search_window_ms=search_window_ms)
    rpeaks = approx_peaks
    return {
        "ecg_bp": ecg_bp,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "approx_peaks": approx_peaks,
        "rpeaks": rpeaks,
        "integrator_window_samples": win_samples,
        "peak_props": props
    }

# --- MAIN ---
if __name__ == "__main__":
    time, ecg, fs = load_ecg(CSV_PATH)
    print(f"Loaded {len(ecg)} samples, fs = {fs:.2f} Hz")
    # run our PT pipeline
    results = run_pan_tompkins_pipeline(time, ecg, fs,
                                        bp_low=5.0, bp_high=15.0,  # możesz próbować 5-35
                                        maw_ms=150,
                                        detect_distance_ms=200,
                                        search_window_ms=80)
    rpeaks_pt = results["rpeaks"]
    print(f"Pan-Tompkins (custom) wykrył {len(rpeaks_pt)} piki.")

    # WYKRES
    fig, ax = plt.subplots(figsize=(12, 8))

    # Wykres z danymi przepuszczonymi przez filtr pasmowy (ecg_bp)
    ax.plot(time, results["ecg_bp"], label=f"ECG bandpass {5}-{15} Hz", alpha=0.9)

    # Oznaczenie szczytów R (custom) na wykresie bandpass
    if rpeaks_pt.size > 0:
        ax.scatter(time[rpeaks_pt], results["ecg_bp"][rpeaks_pt], c='lime', marker='x', s=80,
                   label="R peaks (Pan-Tompkins)")

    # Dodanie etykiet i legendy
    ax.set_title("ECG Bandpass and R-peak Detection")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    # Poprawa układu
    plt.tight_layout()

    # Wyświetlenie wykresu
    plt.show()
