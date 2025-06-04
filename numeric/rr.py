import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d


# PLIK DO ROZPOZNAWANIA SZCZYTOW W SYGNALE

def bandpass_filter(sig, frequency, lowcut: float = 5, highcut: float = 18):
    nyquist = 0.5 * frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def filter_ecg_with_timestamps(ecg_signal, frequency: float):
    timestamps = ecg_signal[:, 0]
    ecg_values = ecg_signal[:, 1]

    filtered_signal = bandpass_filter(ecg_values, frequency, lowcut=0.5, highcut=40)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.150 * frequency)  # 150 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    # Combine timestamps with filtered signal values
    filtered_signal_with_timestamps = np.column_stack((timestamps, integrated_signal))

    return filtered_signal_with_timestamps


def derivative_filter(sig):
    return np.diff(sig, prepend=0)


def square(sig):
    return sig ** 2


def moving_window_integration(sig, window_size):
    return np.convolve(sig, np.ones(window_size), mode="same")


def refine_peak_positions(ecg_signal, detected_peaks, search_window=10):
    refined_peaks = []

    for peak in detected_peaks:
        start = max(peak - search_window, 0)  # Ensure the window doesn't go out of bounds
        end = min(peak + search_window, len(ecg_signal) - 1)

        refined_peak = np.argmax(ecg_signal[start:end]) + start
        refined_peaks.append(refined_peak)

    return np.array(refined_peaks)


def find_r_peaks_values(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal, frequency)

    return ecg_signal[peaks, 1]


def find_r_peaks_values_with_timestamps(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal[:, 1], frequency)

    peak_values = ecg_signal[peaks, 1]
    peak_timestamps = ecg_signal[peaks, 0]

    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))

    return peaks_with_timestamps


def find_r_peaks_ind(ecg_signal, frequency: float):
    filtered_signal = bandpass_filter(ecg_signal, frequency)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.050 * frequency)  # 50 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    clipped_signal = np.clip(integrated_signal, 0, np.percentile(integrated_signal, 99))
    threshold = np.mean(clipped_signal) + 0.6 * np.std(clipped_signal)
    peaks, _ = signal.find_peaks(
        integrated_signal, height=threshold, distance=int(0.4 * frequency)  # 400 ms
    )

    refined_peaks = refine_peak_positions(ecg_signal, peaks, round(10 / 130 * frequency))
    # refined_peaks = processing.correct_peaks(
    #     sig=ecg_signal,
    #     peak_inds=peaks,
    #     search_radius=30,
    #     smooth_window_size=30,
    #     # peak_dir="up",
    # )
    # refined_peaks = peaks

    return refined_peaks

# Calculate the FFT of RR intervals
def fft_rr_intervals(rr_intervals, fs):
    # Apply FFT to the RR intervals
    rr_fft = np.fft.fft(rr_intervals)
    # Calculate the corresponding frequency axis
    freqs = np.fft.fftfreq(len(rr_intervals), d=1/fs)

    # Get the positive frequencies (we usually ignore the negative frequencies)
    positive_freqs = freqs[:len(freqs)//2]
    positive_rr_fft = np.abs(rr_fft)[:len(rr_fft)//2]  # Use absolute value of FFT results

    return positive_freqs, positive_rr_fft


def estimate_breath_rate(rr_intervals, sampling_rate=130):
    # Convert RR intervals to time series
    rr_intervals = np.array(rr_intervals)
    time = np.cumsum(rr_intervals) / sampling_rate  # cumulative sum of RR intervals to get time axis

    # Perform Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(rr_intervals)
    freqs = np.fft.fftfreq(len(rr_intervals), d=1/sampling_rate)  # Frequency axis

    # Only look at the positive frequencies (real component)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_fft = np.abs(fft_result[:len(fft_result) // 2])

    # Find the peak frequency in the low-frequency band
    peak_idx = np.argmax(positive_fft)
    breath_rate = positive_freqs[peak_idx] * 60  # Convert to breaths per minute


    return breath_rate