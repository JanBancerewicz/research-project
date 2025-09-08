import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import torch

from r_neural import get_model, predict
from pan_tompkins import run_pan_tompkins_pipeline

CSV_PATH = "../data/ecg/ECG2.csv"
FS = 130

def run_ai_model(ecg_signal, fs):
    """
    Run the AI model to detect peaks in the ECG signal.
    :param ecg_signal: The raw ECG signal (1D array).
    :param fs: Sampling frequency of the ECG signal.
    :return: Indices of detected peaks.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    model = get_model(device)

    chunk_size = 256  # Adjust based on your model's input size
    num_chunks = len(ecg_signal) // chunk_size
    ecg_chunks = np.array_split(ecg_signal[:num_chunks * chunk_size], num_chunks)

    all_peaks = []
    for i, chunk in enumerate(ecg_chunks):
        chunk = chunk.astype(np.float32)
        peaks = predict(device, model, chunk)
        chunk_peaks = np.where(peaks == 1)[0] + i * chunk_size
        all_peaks.extend(chunk_peaks)

    return np.array(all_peaks)

def run_pan_tompkins_raw(ecg_signal, fs, refine=False):
    """
    Run the Pan-Tompkins algorithm and return indices of R-peaks.
    """
    time = np.arange(len(ecg_signal)) / fs
    result = run_pan_tompkins_pipeline(time, ecg_signal, fs, czy_refine=refine)
    rpeaks = result["rpeaks"]

    # If rpeaks are times, convert to indices
    if np.issubdtype(rpeaks.dtype, np.floating):
        rpeaks = np.array([np.argmin(np.abs(time - t)) for t in rpeaks], dtype=int)

    return rpeaks

def compare_peaks_vs_real(detected_peaks, real_peaks, tolerance):
    """
    Compare detected peaks to real peaks with a given tolerance.
    Returns TP, FP, FN, precision, recall, F1-score.
    """
    tp = 0
    fp = 0
    fn = 0
    matched_real = set()

    for peak in detected_peaks:
        if any(abs(peak - real) <= tolerance for real in real_peaks):
            tp += 1
            real_match = min(real_peaks, key=lambda x: abs(x - peak))
            matched_real.add(real_match)
        else:
            fp += 1

    fn = len(real_peaks) - len(matched_real)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1

def main():
    # 1. Load data
    df = pd.read_csv(CSV_PATH)
    if "ecg" in df.columns:
        ecg_signal = df["ecg"].values
    elif "ECG" in df.columns:
        ecg_signal = df["ECG"].values
    else:
        raise ValueError("CSV file must have column 'ecg' or 'ECG'")

    # 2. AI peak detection
    ai_peaks = run_ai_model(ecg_signal, FS)

    # 3. Pan-Tompkins peak detection
    pt_peaks = run_pan_tompkins_raw(ecg_signal, FS, refine=True)

    # 4. NeuroKit real peaks
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=FS)
    r_peaks = info["ECG_R_Peaks"]

    t = np.arange(len(ecg_signal)) / FS

    # --- Plot: CNN (AI) vs Pan-Tompkins ---
    plt.figure(figsize=(15, 6))
    plt.plot(t, ecg_signal, label="ECG", alpha=0.7)
    plt.scatter(t[ai_peaks], ecg_signal[ai_peaks], color="red", label="AI peaks", marker="o")
    plt.scatter(t[pt_peaks], ecg_signal[pt_peaks], color="green", label="Pan-Tompkins peaks", marker="x")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("AI (CNN) vs Pan-Tompkins peak detection")
    plt.show()

    # --- Plot: CNN (AI) vs Real (NeuroKit) ---
    plt.figure(figsize=(15, 6))
    plt.plot(t, ecg_signal, label="ECG", alpha=0.7)
    plt.scatter(t[ai_peaks], ecg_signal[ai_peaks], color="red", label="AI peaks", marker="o")
    plt.scatter(t[r_peaks], ecg_signal[r_peaks], color="blue", label="Real peaks (NeuroKit)", marker="x")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("AI (CNN) vs Real peaks (NeuroKit)")
    plt.show()

    # --- Statistics ---
    print(f"Number of peaks (AI): {len(ai_peaks)}")
    print(f"Number of peaks (Pan-Tompkins): {len(pt_peaks)}")
    print(f"Number of peaks (real/NeuroKit): {len(r_peaks)}")

    tolerance = int(0.01 * FS)  # 10 ms tolerance at fs=130 Hz ~ 1-2 samples

    # --- Accuracy: AI vs Pan-Tompkins ---
    tp, fp, fn, precision, recall, f1 = compare_peaks_vs_real(ai_peaks, pt_peaks, tolerance)
    print("\n--- Accuracy: AI vs Pan-Tompkins ---")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # --- Accuracy: AI vs real peaks (NeuroKit) ---
    tp_ai, fp_ai, fn_ai, prec_ai, rec_ai, f1_ai = compare_peaks_vs_real(ai_peaks, r_peaks, tolerance)
    print("\n--- Accuracy: AI vs real peaks (NeuroKit) ---")
    print(f"True Positives: {tp_ai}")
    print(f"False Positives: {fp_ai}")
    print(f"False Negatives: {fn_ai}")
    print(f"Precision: {prec_ai:.3f}")
    print(f"Recall:    {rec_ai:.3f}")
    print(f"F1-score:  {f1_ai:.3f}")

    # --- Accuracy: Pan-Tompkins vs real peaks (NeuroKit) ---
    tp_pt, fp_pt, fn_pt, prec_pt, rec_pt, f1_pt = compare_peaks_vs_real(pt_peaks, r_peaks, tolerance)
    print("\n--- Accuracy: Pan-Tompkins vs real peaks (NeuroKit) ---")
    print(f"True Positives: {tp_pt}")
    print(f"False Positives: {fp_pt}")
    print(f"False Negatives: {fn_pt}")
    print(f"Precision: {prec_pt:.3f}")
    print(f"Recall:    {rec_pt:.3f}")
    print(f"F1-score:  {f1_pt:.3f}")

if __name__ == "__main__":
    main()
