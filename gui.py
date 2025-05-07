import asyncio
import threading
import time
from tkinter import ttk

import numpy as np
import pandas as pd
import torch
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from r_neural import get_model

SAMPLE_INTERVAL_MS = 100 / 13
POLAR_NAME = "Polar H10 D222AF24"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

ecg_data = []
lock = threading.Lock()


async def scan_for_device():
    devices = await BleakScanner.discover()
    for dev in devices:
        if dev.name == POLAR_NAME:
            return dev.address
    raise Exception(f"Device '{POLAR_NAME}' not found.")


def handle_ecg_data(_: BleakGATTCharacteristic, data: bytearray):
    if data[0] == 0x00:
        i = 9
        frame_type = data[i]
        if frame_type == 0:
            i += 1
            while len(data[i:]) >= 3:
                with lock:
                    ecg_data.append(int.from_bytes(data[i:i+2], byteorder='little', signed=True))
                i += 3


async def connect_and_stream(address: str):
    async with BleakClient(address) as client:
        print("Connected to", address)
        await client.start_notify(PMD_DATA, handle_ecg_data)
        await client.write_gatt_char(
            PMD_CONTROL,
            bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0e, 0x00])
        )

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await client.write_gatt_char(PMD_CONTROL, bytearray([0x03, 0x00]))
            await client.stop_notify(PMD_DATA)
            print("Disconnected")


def start_ble_thread():
    """Start BLE logic in background thread."""
    def runner():
        asyncio.run(main_async())
    threading.Thread(target=runner, daemon=True).start()


async def main_async():
    address = await scan_for_device()
    await connect_and_stream(address)



def save_csv(ecg, breath):
    timestamp = np.linspace(0, len(ecg)*SAMPLE_INTERVAL_MS, len(ecg))

    # Close off the last breath region
    last_b = breath[-1]
    breath.append([last_b[1], len(ecg), "inhale" if last_b[2] == "exhale" else "exhale"])

    # Build full breath label list
    b = []
    for i in breath:
        for j in range(i[0], i[1]):
            b.append(i[2])

    # Make sure lengths match before saving
    min_len = min(len(timestamp), len(ecg), len(b))
    data = {
        "timestamp": timestamp[:min_len],
        "ecg": ecg[:min_len],
        "breath": b[:min_len],
    }

    df = pd.DataFrame(data)
    df.to_csv("ecg.csv", index=False)

# GUI Class
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.breath_regions = []  # (start_index, end_index, state)
        self.current_breath_state = None
        self.current_breath_start = None
        root.title("Live ECG Viewer")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = get_model(device)

        # --- Button Frame ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start ECG", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop ECG", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.breath_state_v = tk.BooleanVar(value=False)

        self.breath = ttk.Checkbutton(
            btn_frame,
            text="Breath",
            variable=self.breath_state_v,
            style="Toggle.TButton"
        )
        self.breath.pack(side=tk.RIGHT, padx=5)

        # --- Plotting ---
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(-1000, 3500)
        self.ax.set_title("Live ECG")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.line, = self.ax.plot([], [], color='blue', lw=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.window_size = 650
        self.last_index = 0
        self.running = False

    def update_plot(self):
        if not self.running:
            return

        with lock:
            new_data = ecg_data[self.last_index:]

            ##PRINT IF PRESSED DO IT
            new_state = "inhale" if self.breath_state_v.get() else "exhale"
            if new_state != self.current_breath_state:
                now_index = self.last_index
                if self.current_breath_state is not None and self.current_breath_start is not None:
                    self.breath_regions.append((self.current_breath_start, now_index, self.current_breath_state))
                self.current_breath_state = new_state
                self.current_breath_start = now_index

            self.last_index += len(new_data)
            full_data = ecg_data[-self.window_size:] if len(ecg_data) > self.window_size else ecg_data[:]
            # Clear previous vertical lines
        # Remove previous shaded regions
        [p.remove() for p in self.ax.patches]

        # Draw breath regions as shaded backgrounds
        for start_idx, end_idx, state in self.breath_regions:
            start_x = start_idx * SAMPLE_INTERVAL_MS
            end_x = end_idx * SAMPLE_INTERVAL_MS
            color = 'green' if state == "inhale" else 'red'
            self.ax.axvspan(start_x, end_x, facecolor=color, alpha=0.1)

        # Handle current active region (if stream is still ongoing)
        if self.current_breath_start is not None:
            start_x = self.current_breath_start * SAMPLE_INTERVAL_MS
            end_x = self.last_index * SAMPLE_INTERVAL_MS
            color = 'green' if self.current_breath_state == "inhale" else 'red'
            self.ax.axvspan(start_x, end_x, facecolor=color, alpha=0.1)

        if len(full_data) >= 2:
            y = np.array(full_data)
            x = np.arange(len(ecg_data))[-len(y):] * SAMPLE_INTERVAL_MS

            self.line.set_xdata(x)
            self.line.set_ydata(y)
            self.ax.set_xlim(x[0], x[-1])
            self.canvas.draw()

        self.root.after(100, self.update_plot)

    def start_stream(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_plot()

    def stop_stream(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        save_csv(ecg_data, self.breath_regions)



if __name__ == "__main__":
    start_ble_thread()
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()
