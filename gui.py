import asyncio
import queue
import threading
from tkinter import ttk
import numpy as np
import pandas as pd
import websockets
import neurokit2 as nk
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch

from cnn.rmssd.RMSSDModel import get_rmssd_model
from ecg_calc import ECGProcessor
from r_neural import get_model, predict

dane_queue = queue.Queue()

def print_data(y_hr_2):
    try:
        dane = dane_queue.get(timeout=1)
        y_hr_2.append(float(dane))
        if len(y_hr_2) > 100:
            y_hr_2 = y_hr_2[-100:]
        return y_hr_2
    except queue.Empty:
        return y_hr_2


def run_websocket_server():
    async def handler(websocket):
        print("🔌 Connected to client")
        try:
            async for message in websocket:
                dane_queue.put(message)
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed")

    async def start():
        server = await websockets.serve(handler, "localhost", 8765)
        print("✅ Server is listening on port 8765")
        await server.wait_closed()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start())
    loop.run_forever()

SAMPLE_INTERVAL_MS = 100 / 13
POLAR_NAME = "Polar H10 D222AF24"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"


#CHANGE FILE TO SAVE OUTPUT
CSV_DATA = "data5.csv"
rmssd_window_size = 10
ecg_data = []
ecg_window = []
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
                    ecg_window.append(int.from_bytes(data[i:i+2], byteorder='little', signed=True))
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




def save_csv(data):
    columns = ['rmssd', 'sdnn', 'hr', 'edr_mean', 'rr_slope', 'breath_state']

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(CSV_DATA, index=False)

# GUI Class
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.breath_regions = []  # (start_index, end_index, state)
        self.current_breath_state = None
        self.current_breath_start = None
        self.r_peaks = []
        self.r_peaks_times = []
        self.ecg_metadata = []
        self.r_peak_x = []
        self.idx = 0
        self.processor = ECGProcessor()
        root.title("Live ECG Viewer")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = torch.device(device)

        self.model = get_model(self.device)
        self.modelRMSSD = get_rmssd_model(self.device, rmssd_window_size)

        # --- Button Frame ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start ECG", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop ECG", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.breath_state_v = tk.BooleanVar(value=False)

        self.label = tk.Label(root, text="Waiting for data...", font=("Helvetica", 14))
        self.label.pack(pady=10)

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

        self.fig_hr = Figure(figsize=(3, 4))
        self.ax_hr = self.fig_hr.add_subplot(111)
        self.ax_hr.set_xlim(0, 100)
        self.ax_hr.set_ylim(60, 130)
        self.ax_hr.set_title("Live HR")
        self.ax_hr.set_xlabel("Time (ms)")
        self.ax_hr.set_ylabel("bpm")
        self.ax_hr.grid(True)

        self.line_hr1, = self.ax_hr.plot([], [], color='green', lw=1, label='HR POLAR (ai)')
        self.line_hr2, = self.ax_hr.plot([], [], color='blue', lw=1, label='HR APP')
        self.line_hr3, = self.ax_hr.plot([], [], color='red', lw=1, label='HR POLAR (neurokit)')


        self.ax_hr.legend()  # Show legend to differentiate the lines

        self.canvas_hr = FigureCanvasTkAgg(self.fig_hr, master=root)
        self.canvas_hr.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        self.y_hr_1 = []
        self.y_hr_2 = []
        self.y_hr_3 = []

        self.window_size = 650
        self.last_index = 0
        self.running = False
        self.rmssd_peaks = []

        self.r_peaks_for_hr = []
        self.r_peaks_for_hr_neurokit = []

    def update_plot(self):
        if not self.running:
            return
        r_val = []
        r_amp = []
        text = ""
        self.y_hr_2 = print_data(self.y_hr_2)
        self.line_hr2.set_xdata(np.arange(0, len(self.y_hr_2)))
        self.line_hr2.set_ydata(self.y_hr_2)
        self.canvas_hr.draw()

        with lock:
            new_data = ecg_data[self.last_index:]
            if len(ecg_window) >= 256:
                rest = []
                if len(ecg_window) > 256:
                    rest = ecg_window[256:]

                d = predict(self.device, self.model, np.array(ecg_window[:256],  dtype=np.float32))
                self.idx+=1
                if len(self.r_peaks_for_hr) > 0:
                    self.r_peaks_for_hr = [self.r_peaks_for_hr[-1]]
                for idx, s in enumerate(d):
                    if s == 1:
                        t=self.idx*SAMPLE_INTERVAL_MS*len(d) + idx*SAMPLE_INTERVAL_MS
                        r_val.append(t)
                        r_amp.append(ecg_window[:256][idx])
                        self.rmssd_peaks.append(t)
                for i in d:
                    self.r_peaks.append(i)

                for i in r_val:
                    self.r_peaks_for_hr.append(i)
                hr_1 = 60000.0 / np.diff(self.r_peaks_for_hr)

                for i in hr_1:
                    if 50 < i < 120:
                        self.y_hr_1.append(i)
                if len(self.y_hr_1) > 100:
                    self.y_hr_1 = self.y_hr_1[-100:]
                self.line_hr1.set_xdata(np.arange(0, len(self.y_hr_1)))
                self.line_hr1.set_ydata(self.y_hr_1)


                signals, info = nk.ecg_peaks(ecg_window, sampling_rate=130)
                rpeaks = info["ECG_R_Peaks"]
                for peak in rpeaks:
                    self.r_peaks_for_hr_neurokit.append(peak)

                rr_intervals = np.diff(rpeaks) / 130
                bpm_values = 60 / rr_intervals
                for i in bpm_values:
                    if 60 <= i <= 120:
                        self.y_hr_3.append(i)

                print(min(bpm_values), max(bpm_values))
                if len(self.y_hr_3) > 100:
                    self.y_hr_3 = self.y_hr_3[-100:]
                self.line_hr3.set_xdata(np.arange(0, len(self.y_hr_3)))
                self.line_hr3.set_ydata(self.y_hr_3)

                self.canvas_hr.draw()
                ecg_window.clear()
                for i in rest:
                    ecg_window.append(i)
                if len(self.rmssd_peaks) >= 11:
                    rmmsd_p = [np.diff(self.rmssd_peaks[:11])]
                    with torch.no_grad():
                        input_tensor = torch.tensor(rmmsd_p, dtype=torch.float32).to(self.device)
                        predictions = self.modelRMSSD(input_tensor)
                        t = 60000.0 / np.mean(np.diff(r_val))


                        text += f"RMSSD: {predictions.cpu().numpy()[0][0]}\n  HR: {t} bmp"
                        self.label.config(text=text)
                    if len(self.rmssd_peaks) > 11:
                        self.rmssd_peaks = self.rmssd_peaks[-10:]
                    else:
                        self.rmssd_peaks = [self.rmssd_peaks[0]]




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

        if self.current_breath_start is not None:
            start_x = self.current_breath_start * SAMPLE_INTERVAL_MS
            end_x = self.last_index * SAMPLE_INTERVAL_MS
            color = 'green' if self.current_breath_state == "inhale" else 'red'
            self.ax.axvspan(start_x, end_x, facecolor=color, alpha=0.1)

        if len(full_data) >= 2:
            y = np.array(full_data)
            x = np.arange(len(ecg_data))[-len(y):] * SAMPLE_INTERVAL_MS

            # Remove previous dots if present

            r_temp = self.r_peaks.copy()
            for i in range(len(ecg_data) - len(r_temp)):
                r_temp.append(0)
            if hasattr(self, "r_peak_dots"):
                self.r_peak_dots.remove()

            # Get the most recent r_peaks for current visible window
            r_peaks_visible = r_temp[-len(y):] if len(r_temp) >= len(y) else [0] * (
                        len(y) - len(r_temp)) + r_temp


            # Indices of peaks in current window
            r_peak_indices = [i for i, val in enumerate(r_peaks_visible) if val == 1]

            # Get x and y positions of peaks
            self.r_peak_x = [x[i] for i in r_peak_indices]


            r_peak_y = [y[i] for i in r_peak_indices]

            # Plot as red dots
            self.r_peak_dots = self.ax.scatter(self.r_peak_x, r_peak_y, color='red', s=30, zorder=5)

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

        print("--------------------------------")
        print("STREAM STOPPED")
        print("len meta", len(self.ecg_metadata))


        save_csv(self.ecg_metadata)



if __name__ == "__main__":

    start_ble_thread()
    server_socket = threading.Thread(target=run_websocket_server, daemon=True)
    server_socket.start()
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()


