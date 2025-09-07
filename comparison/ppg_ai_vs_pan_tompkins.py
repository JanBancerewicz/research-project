# ppg_ai_vs_pan_tompkins.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pan_tompkins import bandpass, pan_tompkins_transform, detect_qrs_from_integrated
from cnn.ppg.data import get_or_train_model, predict_ppg_segment
from scipy.signal import find_peaks

# domyślny CSV (zmień ścieżkę jeśli trzeba)
# CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data", "ppg_data_1.csv")
# CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data", "ppg_data.csv")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data", "ppg_data_johnny_10min.csv")
# todo poeksperymentowac tutaj z plikami bo cos nie smiga i output zalezy od inputu
# CSV_PATH = os.path.join(os.path.dirname(__file__),"ppg_data_stadarized.csv")

def load_ppg(csv_path):
    """
    Wczytuje CSV z kolumnami (time, ppg).
    Automatycznie wykrywa epoch-ms vs sekundy i zwraca time (sekundy od startu), signal i fs[Hz].
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={df.columns[0]: "time", df.columns[1]: "ppg"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["ppg"]  = pd.to_numeric(df["ppg"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    time = df["time"].values.astype(float)
    signal = df["ppg"].values.astype(float)
    if len(time) < 2:
        raise ValueError("Za mało próbek w pliku PPG")
    dt_raw = np.median(np.diff(time))
    median_time = np.median(time)
    if median_time > 1e9 or dt_raw > 1.0:
        time = (time - time[0]) / 1000.0
    else:
        time = time - time[0]
    dt = np.median(np.diff(time))
    if dt <= 0:
        raise ValueError("Nieprawidłowy krok czasowy po konwersji (dt <= 0)")
    fs = 1.0 / dt
    return time, signal, fs

def run_pan_tompkins_ppg(time, ppg, fs):
    """
    Pan-Tompkins zmodyfikowany pod PPG:
      - filtr 0.5-5 Hz
      - okno integratora ~200 ms
      - minimalna odległość między pikami ~300 ms (dostosujesz)
    Zwraca słownik z ppg_bp i indeksami 'peaks' (indeksy próbek).
    """
    ppg_bp = bandpass(ppg, fs, lowcut=0.5, highcut=5.0, order=3)
    derivative, squared, integrated, win_samples = pan_tompkins_transform(ppg_bp, fs, maw_len_ms=200)
    approx_peaks, props = detect_qrs_from_integrated(integrated, fs,
                                                     distance_ms=300,
                                                     height=None,
                                                     prominence=None)
    return {
        "ppg_bp": ppg_bp,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "peaks": np.asarray(approx_peaks, dtype=int)
    }

def _normalize_window(window):
    min_val = np.min(window)
    max_val = np.max(window)
    if max_val - min_val == 0:
        return np.zeros_like(window)
    return 2 * (window - min_val) / (max_val - min_val) - 1

def run_ai_ppg(signal, fs, model_path=None, segment_length=100):
    """
    Uruchamia model PPG AI w trybie okienkowym.
    Zwraca indeksy próbek, gdzie wykryto piki.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "ppg_peak_model.pth")
    if not os.path.exists(model_path):
        print("[run_ai_ppg] Model not found, falling back to find_peaks.")
        # fallback: klasyczne find_peaks
        sig_bp = bandpass(signal, fs, lowcut=0.5, highcut=5.0, order=3)
        min_dist_samples = max(1, int(round(0.35 * fs)))
        amp_range = np.ptp(sig_bp)
        prom = max(0.15 * amp_range, 0.1)
        peaks, _ = find_peaks(sig_bp, distance=min_dist_samples, prominence=prom)
        print(f"[run_ai_ppg] Fallback detected {len(peaks)} peaks.")
        return np.asarray(peaks, dtype=int)

    try:
        print(f"[run_ai_ppg] Loading model from: {model_path}")
        model = get_or_train_model(model_path=model_path)
        all_peaks = []
        for i in range(0, len(signal), segment_length):
            segment = signal[i:i + segment_length]
            if len(segment) < segment_length:
                break
            normalized_segment = _normalize_window(bandpass(segment, fs, lowcut=0.5, highcut=5.0))
            out = predict_ppg_segment(model, normalized_segment)
            peaks_in_segment = np.where(out > 0.5)[0]
            # Przesunięcie indeksów pików
            adjusted_peaks = peaks_in_segment + i
            all_peaks.extend(adjusted_peaks)
        print(f"[run_ai_ppg] AI detected {len(all_peaks)} peaks in total.")
        return np.asarray(all_peaks, dtype=int)
    except Exception as e:
        print(f"[run_ai_ppg] Problem with model inference, falling back to find_peaks. Error: {e}")
        # fallback: klasyczne find_peaks
        sig_bp = bandpass(signal, fs, lowcut=0.5, highcut=5.0, order=3)
        min_dist_samples = max(1, int(round(0.35 * fs)))
        amp_range = np.ptp(sig_bp)
        prom = max(0.15 * amp_range, 0.1)
        peaks, _ = find_peaks(sig_bp, distance=min_dist_samples, prominence=prom)
        print(f"[run_ai_ppg] Fallback detected {len(peaks)} peaks.")
        return np.asarray(peaks, dtype=int)

def compute_metrics(ai_peaks, pt_peaks, fs, tolerance_ms=150):
    # (reszta funkcji compute_metrics bez zmian)
    tol_samples = int(round(tolerance_ms / 1000.0 * fs))
    ai = np.sort(np.asarray(ai_peaks, dtype=int))
    pt = np.sort(np.asarray(pt_peaks, dtype=int))
    matched_pt = set()
    tp = 0
    fp = 0
    for a in ai:
        diffs = np.abs(pt - a)
        idxs = np.where(diffs <= tol_samples)[0]
        if idxs.size > 0:
            closest_order = idxs[np.argsort(diffs[idxs])]
            found = False
            for idx in closest_order:
                candidate = int(pt[idx])
                if candidate not in matched_pt:
                    matched_pt.add(candidate)
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1
        else:
            fp += 1
    fn = len(pt) - len(matched_pt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "tolerance_samples": tol_samples
    }


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    print(f"Using CSV: {CSV_PATH}")
    time, ppg, fs = load_ppg(CSV_PATH)
    print(f"Loaded PPG: {len(ppg)} samples, fs = {fs:.2f} Hz")

    # Pan-Tompkins (PPG-adapted)
    pt_res = run_pan_tompkins_ppg(time, ppg, fs)
    peaks_pt = pt_res["peaks"]

    # AI (model lub fallback)
    # segment_length = 100 jest domyślną wartością w PPGProcessor, ale możesz dostosować
    peaks_ai = run_ai_ppg(ppg, fs, segment_length=100)

    # Metryki porównawcze
    metrics = compute_metrics(peaks_ai, peaks_pt, fs, tolerance_ms=150)
    print("\n--- Porównanie AI vs Pan-Tompkins (PPG) ---")
    print(f"Pan-Tompkins peaks: {len(peaks_pt)}")
    print(f"AI peaks: {len(peaks_ai)}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
    print(f"(tolerance = {metrics['tolerance_samples']} samples = {metrics['tolerance_samples']/fs*1000:.0f} ms)")

    # Wykres
    plt.figure(figsize=(12, 5))
    plt.plot(time, ppg, label="PPG (raw)", alpha=0.6)
    plt.plot(time, pt_res["ppg_bp"], label="PPG bandpass", alpha=0.6)

    if peaks_pt.size > 0:
        plt.scatter(time[peaks_pt], ppg[peaks_pt], color="green", marker="x", s=60, label="Pan-Tompkins")
    if peaks_ai.size > 0:
        plt.scatter(time[peaks_ai], ppg[peaks_ai], color="red", marker="o", s=40, label="AI")

    plt.title("PPG: AI vs Pan-Tompkins")
    plt.xlabel("Time [s] (relative)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()