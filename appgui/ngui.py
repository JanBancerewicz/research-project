import queue
import threading
import tkinter as tk
from tkinter import ttk
import pandas as pd

from appgui import plot
from appgui.EcgDataFile import EcgDataFile
from appgui.PPGProgessor import PPGProcessor
from appgui.PpgData import PpgData
from appgui.control import ControlPanel
from appgui.ECGProcessor import ECGProcessor


class LiveCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HR Monitor")

        self.ppg_start_time = -1
        self.ppg_out_data = []
        self.ppg_out_time = []
        self.counter = 0
        self.is_running = True

        # Procesory
        self.processorECG = ECGProcessor(window_size=256)
        self.processorPPG = PPGProcessor()

        # Panel kontrolny
        self.controls = ControlPanel(
            root,
            pause_callback=self.pause,
            resume_callback=self.resume,
            reset_callback=self.reset
        )
        self.controls.pack(pady=10)

        # Notebook (zakładki)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="PPG + ECG")
        self.notebook.add(self.tab2, text="HRV")

        # --- Zakładka 1: wykresy ECG i PPG ---
        frame1 = tk.Frame(self.tab1)
        frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        frame2 = tk.Frame(self.tab1)
        frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plotECG = plot.LivePlot(frame1, "ECG", "Time [s]", "Signal")
        self.plotPPG = plot.LivePlot(frame2, "PPG", "Time [s]", "Signal [normalized]", y_lim=(-2, 2), fs=30)
        # --- Zakładka 2: nagłówki kolumn ---
        label_ekg = tk.Label(self.tab2, text="EKG", font=("Arial", 12, "bold"))
        label_ekg.grid(row=0, column=0, padx=5, pady=5)

        label_ppg = tk.Label(self.tab2, text="PPG", font=("Arial", 12, "bold"))
        label_ppg.grid(row=0, column=1, padx=5, pady=5)

        hrv_names = ["rmssd", "sdnn", "pnn50"]
        self.hrv_plots = {}

        for i, name in enumerate(hrv_names):
            row = i + 1

            # EKG (lewa kolumna)
            frame_ekg = tk.Frame(self.tab2)
            frame_ekg.grid(row=row, column=0, sticky="nsew", padx=5, pady=5)
            plot_ekg = plot.LivePlot(frame_ekg, f"EKG - {name.upper()}", "Time [s]", name.upper(), y_lim=(0,100))
            self.hrv_plots[f"ekg_{name}"] = plot_ekg

            # PPG (prawa kolumna)
            frame_ppg = tk.Frame(self.tab2)
            frame_ppg.grid(row=row, column=1, sticky="nsew", padx=5, pady=5)
            plot_ppg = plot.LivePlot(frame_ppg, f"PPG - {name.upper()}", "Time [s]", name.upper())
            self.hrv_plots[f"ppg_{name}"] = plot_ppg

        # Rozciąganie siatki
        for i in range(4):  # 1 label + 3 wiersze z wykresami
            self.tab2.rowconfigure(i, weight=1)
        for j in range(2):
            self.tab2.columnconfigure(j, weight=1)

        # Kolejki i wątki
        self.queueECG = queue.Queue()
        self.queuePPG = queue.Queue()

        self.stop_event_PPG = threading.Event()
        self.stop_event_ECG = threading.Event()

        self.thread1 = EcgDataFile(self.queueECG, self.stop_event_ECG)
        self.thread2 = PpgData(self.queuePPG, self.stop_event_PPG)

        self.thread1.start()
        self.thread2.start()

        self.update_loop()

    def update_loop(self):
        if self.is_running:
            try:
                val1 = self.queueECG.get_nowait()
                self.plotECG.add_data(val1)
                self.counter += 1

                result = self.processorECG.add_sample(val1, (self.counter * (1.0 / 130.0)))
                if result is not None:
                    self.plotECG.add_scatter_points(result.x_peaks, result.y_peaks)
                    rmssd = result.hrv["rmssd"]
                    sdnn = result.hrv["sdnn"]
                    pnn50 = result.hrv["pnn50"]
                    self.hrv_plots["ekg_rmssd"].add_data(rmssd)
                    self.hrv_plots["ekg_sdnn"].add_data(sdnn)
                    self.hrv_plots["ekg_pnn50"].add_data(pnn50)
                    print("HRV:", rmssd, sdnn, pnn50)
            except queue.Empty:
                pass

            try:
                while True:
                    val2 = self.queuePPG.get_nowait()
                    self.ppg_out_data.append(val2[1])
                    self.ppg_out_time.append(val2[0])

                    t = 0
                    if self.ppg_start_time == -1:
                        self.ppg_start_time = self.ppg_out_time[0]
                    else:
                        t = val2[0] - self.ppg_start_time

                    result = self.processorPPG.add_sample(val2[1], t)
                    if result is not None:
                        d = result.filtered_signal
                        for r in range(len(d)):
                            self.plotPPG.add_data(d[r])
                        x_p = [i / 1000.0 for i in result.peak_times]
                        self.plotPPG.add_scatter_points(x_p, result.peak_values)
            except queue.Empty:
                pass

        self.root.after(10, self.update_loop)

    def pause(self):
        self.is_running = False
        df = pd.DataFrame({
            'time': self.ppg_out_time,
            'ppg': self.ppg_out_data,
        })
        df.to_csv('ppg_data.csv', index=False)

    def resume(self):
        self.is_running = True

    def reset(self):
        self.plotPPG.reset()
        self.plotECG.reset()
        for plot_obj in self.analysis_plots:
            plot_obj.reset()

    def on_closing(self):
        self.stop_event_PPG.set()
        self.stop_event_ECG.set()
        self.thread1.join(timeout=1)
        self.thread2.join(timeout=1)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
