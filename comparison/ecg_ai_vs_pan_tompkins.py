import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from r_neural import get_model, predict
from pan_tompkins import run_pan_tompkins_pipeline  # zakładam, że masz plik z funkcją
from pan_tompkins import detect_qrs_from_integrated  # jeśli chcesz czystą wersję

CSV_PATH = "../data/ecg/ECG2.csv"
FS = 130  # Hz, dostosuj do swojego pliku

def run_ai_model(ecg_signal, fs):
    """
    Run the AI model to detect peaks in the ECG signal.
    :param ecg_signal: The raw ECG signal (1D array).
    :param fs: Sampling frequency of the ECG signal.
    :return: Indices of detected peaks.
    """
    # Load the model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    model = get_model(device)

    # Preprocess the signal (split into chunks)
    chunk_size = 256  # Adjust based on your model's input size
    num_chunks = len(ecg_signal) // chunk_size
    ecg_chunks = np.array_split(ecg_signal[:num_chunks * chunk_size], num_chunks)

    # Predict peaks for each chunk
    all_peaks = []
    for i, chunk in enumerate(ecg_chunks):
        chunk = chunk.astype(np.float32)
        peaks = predict(device, model, chunk)

        chunk_peaks = np.where(peaks == 1)[0] + i * chunk_size  # Adjust indices
        all_peaks.extend(chunk_peaks)

    return np.array(all_peaks)



def run_pan_tompkins_raw(ecg_signal, fs):
    """
    Uruchamia algorytm Pan-Tompkinsa i zwraca indeksy R-peaków (int).
    """
    time = np.arange(len(ecg_signal)) / fs
    result = run_pan_tompkins_pipeline(time, ecg_signal, fs)

    # Wyciągamy rpeaks i upewniamy się, że to są indeksy
    rpeaks = result["rpeaks"]

    # Jeśli ktoś przypadkiem zwraca czasy, konwertujemy z powrotem na indeksy
    if np.issubdtype(rpeaks.dtype, np.floating):
        rpeaks = np.array([np.argmin(np.abs(time - t)) for t in rpeaks], dtype=int)

    return rpeaks


def main():
    # --- 1. Wczytanie danych ---
    df = pd.read_csv(CSV_PATH)
    if "ecg" in df.columns:
        ecg_signal = df["ecg"].values
    elif "ECG" in df.columns:
        ecg_signal = df["ECG"].values
    else:
        raise ValueError("Plik CSV musi mieć kolumnę 'ecg' albo 'ECG'")

    # --- 2. AI ---
    ai_peaks = run_ai_model(ecg_signal, FS)

    # --- 3. Pan–Tompkins ---
    pt_peaks = run_pan_tompkins_raw(ecg_signal, FS)

    # --- 4. Rysowanie ---
    t = np.arange(len(ecg_signal)) / FS
    plt.figure(figsize=(15, 6))
    plt.plot(t, ecg_signal, label="ECG", alpha=0.7)
    plt.scatter(t[ai_peaks], ecg_signal[ai_peaks], color="red", label="AI peaks", marker="o")
    plt.scatter(t[pt_peaks], ecg_signal[pt_peaks], color="green", label="Pan-Tompkins peaks", marker="x")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("AI vs Pan-Tompkins peak detection")
    plt.show()

    # --- 5. Statystyki ---
    print(f"Liczba peaków (AI): {len(ai_peaks)}")
    print(f"Liczba peaków (Pan-Tompkins): {len(pt_peaks)}")

    # --- 6. Dokładność porównania ---
    tolerance = int(0.05 * FS)  # 50 ms tolerancji przy fs=130 Hz ~ 6 próbek

    tp = 0  # true positives (AI peak pokrył się z PT)
    fp = 0  # false positives (AI peak nie ma odpowiednika w PT)
    fn = 0  # false negatives (PT peak nie ma odpowiednika w AI)

    matched_pt = set()

    for ai_peak in ai_peaks:
        # sprawdzamy, czy istnieje peak PT w oknie tolerancji
        if any(abs(ai_peak - pt) <= tolerance for pt in pt_peaks):
            tp += 1
            # znajdź pierwszy pasujący peak PT i oznacz go jako użyty
            pt_match = min(pt_peaks, key=lambda x: abs(x - ai_peak))
            matched_pt.add(pt_match)
        else:
            fp += 1

    # policz brakujące peak'i PT
    fn = len(pt_peaks) - len(matched_pt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Dokładność AI vs Pan-Tompkins ---")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


if __name__ == "__main__":
    main()
