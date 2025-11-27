import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks

# Import model
try:
    from cnn.ppg.data import get_or_train_model
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cnn.ppg.data import get_or_train_model

# --- CONFIGURATION ---
ECG_ALIGNED_PATH = "ecg_data_aligned.csv"
PPG_ALIGNED_PATH = "ppg_data_aligned.csv"
MODEL_PATH = "../ppg_peak_model.pth"
TOLERANCE_MS = 250


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def load_aligned_data():
    if not os.path.exists(ECG_ALIGNED_PATH) or not os.path.exists(PPG_ALIGNED_PATH):
        raise FileNotFoundError("Missing aligned files in 'comparison/' folder.")
    print(f"Loading aligned data...")
    df_ecg = pd.read_csv(ECG_ALIGNED_PATH)
    df_ppg = pd.read_csv(PPG_ALIGNED_PATH)
    df_ecg['time'] = pd.to_numeric(df_ecg['time'], errors='coerce')
    df_ppg['time'] = pd.to_numeric(df_ppg['time'], errors='coerce')
    return df_ecg, df_ppg


def get_ecg_ground_truth_peaks(df_ecg):
    print("Detecting Ground Truth R-peaks (NeuroKit)...")
    ecg_signal = df_ecg['ecg'].values
    fs_est = 1000 / np.median(np.diff(df_ecg['time'].values))
    print(f"Estimated ECG sampling rate: {fs_est:.2f} Hz")

    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs_est, method="neurokit")
    signals, info = nk.ecg_peaks(cleaned, sampling_rate=fs_est, method="neurokit")
    peak_times = df_ecg['time'].iloc[info["ECG_R_Peaks"]].values
    return peak_times


def run_ppg_model_probabilistic(df_ppg, model_path):
    print("Running PPG Model (Probabilistic Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs_est = 1000 / np.median(np.diff(df_ppg['time'].values))
    print(f"Estimated PPG sampling rate: {fs_est:.2f} Hz")

    # 1. Filtering (CRITICAL) + INVERSION
    raw_signal = df_ppg['ppg'].values
    raw_signal = raw_signal * -1  # Invert signal (peaks become valleys)
    filtered_signal = butter_bandpass_filter(raw_signal, 0.5, 5.0, fs=fs_est, order=3)

    # 2. Load Model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data.py is in cnn/ppg relative to project root
    # We are in comparison/, so root is one level up
    project_root = os.path.dirname(current_dir)
    dummy_data_dir = os.path.join(project_root, "cnn", "ppg", "train_data")

    # Handle model path relative to script if needed
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "ppg_peak_model.pth")

    model = get_or_train_model(model_path=model_path, data_dir=dummy_data_dir, segment_length=100, epochs=0)
    model.eval()

    # 3. Continuous Prediction
    full_probabilities = np.zeros(len(filtered_signal))
    segment_len = 100
    stride = 100

    with torch.no_grad():
        for i in range(0, len(filtered_signal) - segment_len, stride):
            seg_sig = filtered_signal[i: i + segment_len]

            min_val = np.min(seg_sig)
            max_val = np.max(seg_sig)
            if max_val - min_val == 0:
                norm_seg = np.zeros_like(seg_sig)
            else:
                norm_seg = 2 * (seg_sig - min_val) / (max_val - min_val) - 1

            input_tensor = torch.tensor(norm_seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(input_tensor).squeeze().cpu().numpy()

            end_idx = min(i + segment_len, len(full_probabilities))
            len_to_copy = end_idx - i
            full_probabilities[i: end_idx] = output[:len_to_copy]

    # 4. Smart Peak Detection
    # Lower threshold to improve Recall
    min_dist = int(0.3 * fs_est)
    peak_indices, _ = find_peaks(full_probabilities, height=0.25, distance=min_dist)

    detected_peak_times = df_ppg['time'].iloc[peak_indices].values

    return detected_peak_times, filtered_signal, full_probabilities


def evaluate_peaks(pred_times, true_times, tolerance_ms):
    tp, fp = 0, 0
    matched_indices = set()
    pred_times = np.sort(pred_times)
    true_times = np.sort(true_times)

    for pred in pred_times:
        diffs = np.abs(true_times - pred)
        if len(diffs) == 0: continue
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance_ms:
            if min_idx not in matched_indices:
                tp += 1
                matched_indices.add(min_idx)
            else:
                fp += 1
        else:
            fp += 1
    fn = len(true_times) - len(matched_indices)
    return tp, fp, fn


def main():
    df_ecg, df_ppg = load_aligned_data()
    true_peaks_ecg = get_ecg_ground_truth_peaks(df_ecg)

    pred_peaks_ppg, ppg_filtered, probs = run_ppg_model_probabilistic(df_ppg, MODEL_PATH)

    print(f"Detected PPG peaks: {len(pred_peaks_ppg)}")

    # Auto-correct PTT (Pulse Transit Time)
    diffs = []
    for t in true_peaks_ecg:
        if len(pred_peaks_ppg) == 0: break
        closest = pred_peaks_ppg[np.argmin(np.abs(pred_peaks_ppg - t))]
        if closest >= t and (closest - t) < 500:
            diffs.append(closest - t)
    avg_ptt = np.median(diffs) if diffs else 0
    print(f"Estimated PTT (lag): {avg_ptt:.2f} ms")

    pred_corrected = pred_peaks_ppg - avg_ptt

    tp, fp, fn = evaluate_peaks(pred_corrected, true_peaks_ecg, TOLERANCE_MS)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== VALIDATION RESULT (PPG vs ECG) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # --- PLOTTING (Combined Figure) ---
    plt.figure(figsize=(12, 6))

    # Show first 10 seconds of data (skipping first 2s startup)
    t_start = df_ecg['time'].iloc[0] + 2000
    t_end = t_start + 10000

    mask_ecg = (df_ecg['time'] >= t_start) & (df_ecg['time'] <= t_end)
    mask_ppg = (df_ppg['time'] >= t_start) & (df_ppg['time'] <= t_end)

    # Normalize signals for visual comparison
    ecg_sig = df_ecg.loc[mask_ecg, 'ecg']
    ecg_sig = (ecg_sig - ecg_sig.mean()) / ecg_sig.std()

    ppg_view = ppg_filtered[mask_ppg]
    ppg_view = (ppg_view - ppg_view.mean()) / ppg_view.std()

    # Plot signals
    plt.plot(df_ecg.loc[mask_ecg, 'time'], ecg_sig, label="ECG Reference (Norm)", alpha=0.6, color='royalblue',
             linewidth=1.5)
    plt.plot(df_ppg.loc[mask_ppg, 'time'], ppg_view, label="PPG Filtered (Norm)", alpha=0.8, color='darkorange',
             linewidth=1.5)

    # Plot Peaks
    p_ecg = true_peaks_ecg[(true_peaks_ecg >= t_start) & (true_peaks_ecg <= t_end)]
    p_ppg = pred_corrected[(pred_corrected >= t_start) & (pred_corrected <= t_end)]

    # Draw markers slightly above signal
    y_marker = 2.5
    plt.scatter(p_ecg, [y_marker] * len(p_ecg), color='blue', marker='v', s=60, label='ECG R-Peak', zorder=5)
    plt.scatter(p_ppg, [y_marker + 0.3] * len(p_ppg), color='red', marker='o', s=60, label='PPG Detected Peak',
                zorder=5)

    # Draw vertical lines connecting peaks for visual alignment check
    for p in p_ecg:
        plt.axvline(x=p, color='blue', linestyle=':', alpha=0.2)

    plt.title(f"PPG vs ECG Peak Detection (F1-Score: {f1:.2f})", fontsize=14)
    plt.xlabel("Time [ms]", fontsize=12)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("validation_result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()