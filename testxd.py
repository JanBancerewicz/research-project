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

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import neurokit2 as nk
# Przykładowy (symulowany) sygnał PPG z szumem
fs = 100  # częstotliwość próbkowania (Hz)
ppg_raw = nk.ppg_simulate(10, fs)

# Zastosowanie filtru Savitzky-Golaya
window_length = 51  # musi być nieparzysta i > polyorder
polyorder = 3       # rząd wielomianu
ppg_filtered = savgol_filter(ppg_raw, window_length, polyorder)

# Wykres
plt.figure(figsize=(12, 5))
plt.plot(ppg_raw, label='Sygnał PPG (surowy)', alpha=0.5)
plt.plot(ppg_filtered, label='Sygnał PPG (po filtrze Savitzky-Golaya)', linewidth=2)
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda')
plt.title('Filtracja sygnału PPG - Savitzky-Golay')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()