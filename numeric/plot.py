
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from dirtycode import RR
import scipy.signal as signal
from scipy.interpolate import splrep, splev

from numeric.rr import filter_ecg_with_timestamps, find_r_peaks_values_with_timestamps, fft_rr_intervals

def high_pass_filter(rr_intervals, cutoff_freq=0.1, sampling_rate = 130):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalize the cutoff frequency
    b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)  # Butterworth high-pass filter
    return signal.filtfilt(b, a, rr_intervals)


def interp_cubic_spline(rri, sf_up=4):
    """
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.

    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.

    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
    rri_time = np.cumsum(rri) / 1000.0
    time_rri = rri_time - rri_time[0]
    time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
    tck = splrep(time_rri, rri, s=0)
    rri_interp = splev(time_rri_interp, tck, der=0)
    return rri_interp

def plot(data):
    timestamps = data["timestamp"]  # 10 seconds of ECG signal sampled at 1000 Hz
    ecg_values = data["ecg"]  # Example synthetic ECG signal

    # Sampling frequency
    fs = 130  # 1000 Hz
    ecg_signal = np.column_stack((timestamps, ecg_values))

    f = filter_ecg_with_timestamps(ecg_signal, fs)

    # 2. Find R-peaks with timestamps
    r_peaks_with_timestamps = find_r_peaks_values_with_timestamps(f, fs)

    # 3. Plot the ECG signal with detected R-peaks

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, f[:,1], label="ECG Signal", color="b")



    plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:,1], "ro", label="Detected R-peaks")
    plt.title("ECG Signal with Detected R-peaks")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(r_peaks_with_timestamps[:, 0], r_peaks_with_timestamps[:, 1], label="R-peaks")
    plt.title("R")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    p = r_peaks_with_timestamps[:, 0]
    p /= 130
    x = np.diff(p)
    x = interp_cubic_spline(x, 4)



    # 4. Calculate FFT of RR intervals
    freqs, rr_fft = fft_rr_intervals(x, fs)
    plt.subplot(3, 1, 3)
    plt.title("R-R line")
    plt.plot(x, label="R-R")
    plt.legend()

    plt.tight_layout()

    heart_rate = 1000 *60 / x
    plt.figure(figsize=(10, 6))
    plt.plot(heart_rate, label="HR")
    plt.title("Heart Rate")
    plt.xlabel("Time [s]")
    plt.ylabel("bmp")
    plt.legend()


    # 5. Plot the FFT of RR intervals
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, rr_fft, label="FFT of RR intervals")
    plt.title("FFT of RR Intervals")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()

    print(len(x))
    p = 0


    interval_duration = 3 # seconds

    # Calculate how many RR intervals correspond to 3 seconds (3 seconds * sampling_rate)
    intervals_per_group = interval_duration * 100

    # Create a list to store the breathing rate for each 3-second interval
    breathing_rates = []

    # Split the data into chunks of 3 seconds, calculate the breathing rate for each chunk
    for i in range(0, len(x), intervals_per_group):
        chunk = x[i:i + intervals_per_group]
        if len(chunk) == intervals_per_group:
            filtered_chunk = high_pass_filter(chunk, cutoff_freq=0.1)

            breathing_rate = RR.calculate_breathing_rate_from_RR(filtered_chunk)
            if breathing_rate is not None:
                breathing_rates.append(breathing_rate)

    # Calculate the average breathing rate across the chunks
    average_breathing_rate = np.mean(breathing_rates) if breathing_rates else None

    # Print the results
    if average_breathing_rate is not None:
        print(f"Average Breathing Rate: {average_breathing_rate:.2f} breaths per minute")
    else:
        print("No significant respiratory frequency detected across the intervals.")
    plt.show()
    # start = 0
    # end = 325
    # start = 0
    # RR.calculate_breathing_rate_from_RR(x)
    # b = []
    # while end < len(x):
    #     b.append(RR.calculate_breathing_rate_from_RR(x[start:end]))
    #     end += 20
    #     start += 20
    # plt.figure()
    # plt.plot(b, label="Breathing Rate")
    # plt.show()

