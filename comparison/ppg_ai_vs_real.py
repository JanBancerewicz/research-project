# ppg_ai_vs_pan_tompkins.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pan_tompkins import bandpass, pan_tompkins_transform, detect_qrs_from_integrated
from cnn.ppg.data import get_or_train_model, predict_ppg_segment
from scipy.signal import find_peaks
import torch

# domyślny CSV (zmień ścieżkę jeśli trzeba)
# CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data","ppg_data.csv")
# CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data","ppg_data_1.csv")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comparison", "ppg_data_aligned.csv") # jedyny niezcorruptowany plik xd
# CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data","ppg_data_johnny_10min.csv")



def load_ppg(csv_path):
    """
    Wczytuje CSV z kolumnami (time, ppg).
    Automatycznie wykrywa epoch-ms vs sekundy i zwraca time (sekundy od startu), signal i fs[Hz].
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={df.columns[0]: "time", df.columns[1]: "ppg"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["ppg"] = pd.to_numeric(df["ppg"], errors="coerce")
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


def run_detect_ppg(time, ppg, fs):
    """
    Pan-Tompkins zmodyfikowany pod PPG:
      - filtr 0.5-5 Hz
      - okno integratora ~200 ms
      - minimalna odległość między pikami ~300 ms (dostosujesz)
    Zwraca słownik z ppg_bp i indeksami 'peaks' (indeksy próbek).
    """
    ppg_bp = bandpass(ppg, fs, lowcut=0.5, highcut=5.0, order=3)
    # Zmiana: Pan-Tompkins ma problem z detekcją pików w PPG.
    # Alternatywnie, użyj standardowego find_peaks po przefiltrowaniu sygnału,
    # co często daje lepsze rezultaty dla PPG.
    # Piki PPG są zwykle szczytami, więc szukamy lokalnych maksimów.
    distance_samples = int(round(0.35 * fs))  # 350 ms
    peaks, _ = find_peaks(ppg_bp, distance=distance_samples, prominence=np.std(ppg_bp) * 0.5)

    return {
        "ppg_bp": ppg_bp,
        "peaks": np.asarray(peaks, dtype=int)
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
    :param model_path: Ścieżka do wytrenowanego pliku modelu.
    """
    # Sprawdzenie, czy ścieżka do modelu została podana i czy plik istnieje
    if not model_path or not os.path.exists(model_path):
        print(f"[run_ai_ppg] Model not found at '{model_path}'. Falling back to find_peaks.")
        # fallback: klasyczne find_peaks, używając tych samych parametrów co w "Pan-Tompkins"
        sig_bp = bandpass(signal, fs, lowcut=0.5, highcut=5.0, order=3)
        min_dist_samples = int(round(0.35 * fs))
        peaks, _ = find_peaks(sig_bp, distance=min_dist_samples, prominence=np.std(sig_bp) * 0.5)
        print(f"[run_ai_ppg] Fallback detected {len(peaks)} peaks.")
        return np.asarray(peaks, dtype=int)

    try:
        print(f"[run_ai_ppg] Loading model from: {model_path}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model = get_or_train_model(model_path=model_path).to(device)

        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn", "ppg", "train_data")
        MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"ppg_peak_model.pth")
        SEGMENT_LENGTH = 100 # todo verify window size = 100?
        MAX_SEGMENTS = 10000
        EPOCHS = 200
        BATCH_SIZE = 32
        LR = 0.001
        MAX_FILES = None

        model = get_or_train_model(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            segment_length=SEGMENT_LENGTH,
            max_segments=MAX_SEGMENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_files=MAX_FILES
        ).to(device)

        print(f"[run_ai_ppg] model smiga na: {device}")

        all_peaks = []
        for i in range(0, len(signal), segment_length):
            segment = signal[i:i + segment_length]
            if len(segment) < segment_length:
                break

            # Normalizacja każdego okna (segmentu)
            normalized_segment = _normalize_window(segment).astype(np.float32)
            tensor_segment = torch.from_numpy(normalized_segment).unsqueeze(0).unsqueeze(0).to(device)

            # Predykcja
            out = model(tensor_segment)
            out = out.cpu().detach().numpy().flatten()

            # Wyszukanie pików z predykcji
            peaks_in_segment = np.where(out > 0.5)[0]
            adjusted_peaks = peaks_in_segment + i
            all_peaks.extend(adjusted_peaks)

        print(f"[run_ai_ppg] AI model detected {len(all_peaks)} peaks in total.")
        return np.array(all_peaks)

    except Exception as e:
        print(f"[run_ai_ppg] Problem with model inference. Error: {e}")
        return np.array([])


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
    pt_res = run_detect_ppg(time, ppg, fs)
    peaks_pt = pt_res["peaks"]

    # AI (model or fallback)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"ppg_peak_model.pth")
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: AI model file not found at: {MODEL_PATH}")
        print("AI model will not be used.")
        peaks_ai = np.array([])
    else:
        peaks_ai = run_ai_ppg(ppg, fs, model_path=MODEL_PATH, segment_length=100)

    # Metrics
    metrics = compute_metrics(peaks_ai, peaks_pt, fs, tolerance_ms=150)
    print("Number of peaks (AI):", len(peaks_ai))
    print("Number of peaks (Validated/Reference):", len(peaks_pt))
    print("\n--- Accuracy: AI vs Real ---")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1']:.3f}")
    print(f"(tolerance = {metrics['tolerance_samples']} samples = {metrics['tolerance_samples'] / fs * 1000:.0f} ms)")

    # Normalize PPG in segments
    segment_length = 100
    normalized_ppg = np.zeros_like(ppg)
    for i in range(0, len(ppg), segment_length):
        segment = ppg[i:i+segment_length]
        norm_segment = _normalize_window(segment)
        normalized_ppg[i:i+len(norm_segment)] = norm_segment

    # Plot normalized PPG with peaks
    plt.figure(figsize=(12, 5))
    plt.plot(time, normalized_ppg, label="PPG (normalized, segment-wise)", alpha=0.7)
    if peaks_pt.size > 0:
        plt.scatter(time[peaks_pt], normalized_ppg[peaks_pt], color="green", marker="x", s=60, label="Reference Peaks for PPG")
    if peaks_ai.size > 0:
        plt.scatter(time[peaks_ai], normalized_ppg[peaks_ai], color="red", marker="o", s=40, label="AI Detected Peaks (PPG)")
    plt.title("PPG (normalized, segment-wise): CNN-detected vs Validated Peaks")
    plt.xlabel("Time [s] (relative)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()