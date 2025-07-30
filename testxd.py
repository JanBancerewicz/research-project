import asyncio
import websockets
import random

async def send_random_data():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            while True:
                value = random.randint(0, 2000)
                await websocket.send(str(value))
                print(f"📤 Sent: {value}")
                await asyncio.sleep(0.01)  # 100 ms
    except Exception as e:
        print(f"❌ Connection failed or lost: {e}")

#if __name__ == "__main__":
    #asyncio.run(send_random_data())

import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def normalize_signal(signal):
    """
    Normalize a signal to the range [-1, 1].
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal  # Avoid division by zero
    return 2 * (signal - min_val) / (max_val - min_val) - 1

def simulate_and_get_rr(sampling_rate, duration, heart_rate, respiratory_rate, noise=0.01):
    """
    Simulate ECG and return RR intervals and their times.
    """
    ecg = nk.ecg_simulate(duration=duration,
                          sampling_rate=sampling_rate,
                          heart_rate=heart_rate,
                          respiratory_rate=respiratory_rate,
                          noise=noise)
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]
    rr_times = [rpeaks[i] / sampling_rate for i in range(len(rpeaks) - 1)]
    rr_values = [(rpeaks[i+1] - rpeaks[i]) / sampling_rate for i in range(len(rpeaks) - 1)]
    return rr_times, rr_values

# Test for different HR and RR
sampling_rate = 130
duration = 60  # seconds

params = [
    {"heart_rate": 56, "respiratory_rate": 30, "label": "HR=56, RR=30"},
    {"heart_rate": 80, "respiratory_rate": 18, "label": "HR=80, RR=18"},
    {"heart_rate": 100, "respiratory_rate": 12, "label": "HR=100, RR=12"},
]

plt.figure(figsize=(12, 5))
for p in params:
    rr_times, rr_values = simulate_and_get_rr(
        sampling_rate=sampling_rate,
        duration=duration,
        heart_rate=p["heart_rate"],
        respiratory_rate=p["respiratory_rate"]
    )
    plt.plot(rr_times, normalize_signal(rr_values), marker='o', linestyle='-', label=p["label"])

plt.title("RSA – RR-intervals for different HR and respiratory rates")
plt.xlabel("Time [s]")
plt.ylabel("RR interval [s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
