import queue
import threading
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np

from appgui import plot
from appgui.CompareProcessor import CompareProcessor
from appgui.EcgDataFile import EcgDataFile
from appgui.EcgData import EcgDataBluetooth
from appgui.PPGProgessor import PPGProcessor
from appgui.PpgData import PpgData
from appgui.control import ControlPanel
from appgui.ECGProcessor import ECGProcessor
from data_processing import save_data


class LiveCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HR Monitor")

        # --- Top bar for save buttons ---
        topbar = tk.Frame(root)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        btn_save_ecg = tk.Button(topbar, text="Save ECG", command=self.save_ecg)
        btn_save_ecg.pack(side=tk.LEFT, padx=5)
        btn_save_ppg = tk.Button(topbar, text="Save PPG", command=self.save_ppg)
        btn_save_ppg.pack(side=tk.LEFT, padx=5)
        btn_save_both = tk.Button(topbar, text="Save Both (Aligned)", command=self.save_both)
        btn_save_both.pack(side=tk.LEFT, padx=5)
        # Add button to save all HRV plots as PNG
        btn_save_hrv_png = tk.Button(topbar, text="Save HRV Plots (PNG, 300dpi)", command=self.save_hrv_plots_png)
        btn_save_hrv_png.pack(side=tk.LEFT, padx=5)

        self.ppg_start_time = -1
        self.ppg_out_data = []
        self.ppg_out_time = []
        self.counter = 0
        self.is_running = True

        # Procesory
        self.processorECG = ECGProcessor(window_size=256)
        self.processorPPG = PPGProcessor()
        self.compareProcessor = CompareProcessor()

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
        self.tab3 = ttk.Frame(self.notebook)  # New tab for peaks compare

        self.notebook.add(self.tab1, text="PPG + ECG")
        self.notebook.add(self.tab2, text="HRV")
        self.notebook.add(self.tab3, text="Peaks Compare")  # Add new tab

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

        hrv_names = ["rmssd", "sdnn", "rr"]
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

        # --- Zakładka 3: Peaks Compare ---
        frame_peaks = tk.Frame(self.tab3)
        frame_peaks.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plotPeaksCompare = plot.LivePlot(
            frame_peaks, "Peaks Compare", "Time [s]", "Amplitude"
        )
        # Add a plot for diffs (below or overlay)
        frame_diff = tk.Frame(self.tab3)
        frame_diff.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)


        # Kolejki i wątki
        self.queueECG = queue.Queue()
        self.queuePPG = queue.Queue()

        self.stop_event_PPG = threading.Event()
        self.stop_event_ECG = threading.Event()

        self.thread1 = EcgDataBluetooth(self.queueECG, self.stop_event_ECG)
        # self.thread1 = EcgDataFile(self.queueECG, self.stop_event_ECG)
        self.thread2 = PpgData(self.queuePPG, self.stop_event_PPG)

        self.thread1.start()
        self.thread2.start()

        self.ecg_out_data = []
        self.ecg_out_time = []

        self.update_loop()

    def update_loop(self):
        if self.is_running:
            try:
                while True:
                    val1 = self.queueECG.get_nowait()
                    # print(f"ECG: Timestamp {round(val1[0])}")
                    self.plotECG.add_data(val1[1])
                    self.counter += 1

                    # Add data to ECG output arrays
                    self.ecg_out_time.append(val1[0])
                    self.ecg_out_data.append(val1[1])

                    result = self.processorECG.add_sample(val1[1],val1[0], (self.counter * (1.0 / 130.0)))
                    if result is not None:
                        self.plotECG.add_scatter_points(result.x_peaks, result.y_peaks)
                        self.compareProcessor.add_ecg_peaks(result.peak_unix_times)
                        rmssd = result.hrv["rmssd"]
                        sdnn = result.hrv["sdnn"]
                        # pnn50 = result.hrv["pnn50"]
                        rr_intervals = result.hrv["rr_intervals"]
                        self.hrv_plots["ekg_rmssd"].add_data(rmssd)
                        self.hrv_plots["ekg_sdnn"].add_data(sdnn)
                        for i in rr_intervals:
                          self.hrv_plots["ekg_rr"].add_data(i)
                        self.hrv_plots["ekg_sdnn"].add_data(sdnn)
                        # self.hrv_plots["ekg_pnn50"].add_data(pnn50)
                        # print("HRV:", sdnn, pnn50,  rr_intervals)
            except queue.Empty:
                pass

            try:
                while True:
                    val2 = self.queuePPG.get_nowait()
                    self.ppg_out_data.append(val2[1])


                    t = 0
                    self.ppg_out_time.append(val2[0])
                    if self.ppg_start_time == -1:
                        self.ppg_start_time = self.ppg_out_time[0]
                    else:
                        t = val2[0] - self.ppg_start_time
                    result_tuple = self.processorPPG.add_sample(val2[1], t, val2[0])
                    if result_tuple is not None:
                        result, hrv = result_tuple
                        if result is not None:
                            d = result.filtered_signal
                            for r in range(len(d)):
                                self.plotPPG.add_data(d[r])
                            x_p = [i / 1000.0 for i in result.peak_times]
                            self.plotPPG.add_scatter_points(x_p, result.peak_values)
                            self.compareProcessor.add_ppg_peaks(result.peak_unix_times)
                            # Calculate HRV for PPG
                            # hrv = self.processorPPG.calculate_hrv()
                            if hrv:
                                self.hrv_plots["ppg_rmssd"].add_data(hrv["rmssd"])
                                self.hrv_plots["ppg_sdnn"].add_data(hrv["sdnn"])
                                for rr in hrv["rr_intervals"]:
                                    self.hrv_plots["ppg_rr"].add_data(rr / 1000.0)
                                print("PPG HRV:", hrv["rmssd"], hrv["sdnn"], hrv["rr_intervals"])
            except queue.Empty:
                pass

            # Update the diff plot with latest diffs
            self.update_compare_plot()

        self.root.after(10, self.update_loop)

    def update_compare_plot(self):
        # Plot the diffs as a line or scatter
        diffs = self.compareProcessor.diff
        if diffs:
            x = list(range(len(diffs)))
            self.plotPeaksCompare.set_data(x, diffs)

    def pause(self):
        self.is_running = False
        # Save PPG data
        df = pd.DataFrame({
            'time': self.ppg_out_time,
            'ppg': self.ppg_out_data,
        })
        df.to_csv('ppg_data.csv', index=False)

        # Save ECG data
        ecg_data = list(zip(self.processorECG.timestamps, self.processorECG.ecg_values))
        save_data(ecg_data, 'ecg_data.csv')  # Call save_data function

    def resume(self):
        self.is_running = True

    def reset(self):
        self.plotPPG.reset()
        self.plotECG.reset()

    def save_ecg(self):
        df = pd.DataFrame({
            'time': self.ecg_out_time,
            'ecg': self.ecg_out_data,
        })
        df.to_csv('ecg_data.csv', index=False)
        print("ECG data saved to ecg_data.csv")

    def save_ppg(self):
        df = pd.DataFrame({
            'time': self.ppg_out_time,
            'ppg': self.ppg_out_data,
        })
        df.to_csv('ppg_data.csv', index=False)
        print("PPG data saved to ppg_data.csv")

    def save_both(self):
        # Find the earliest matching timestamp (within +/- 500 ms)
        if not self.ecg_out_time or not self.ppg_out_time:
            print("No ECG or PPG data to save.")
            return

        ecg_times = np.array(self.ecg_out_time)
        ppg_times = np.array(self.ppg_out_time)

        # Find the first pair of indices where timestamps are within 500 ms
        found = False
        for i, t_ecg in enumerate(ecg_times):
            close_ppg = np.where(np.abs(ppg_times - t_ecg) <= 0.5)[0]
            if close_ppg.size > 0:
                idx_ecg = i
                idx_ppg = close_ppg[0]
                found = True
                break

        if not found:
            print("No matching timestamps within 500 ms found.")
            return

        # Prepare peak times sets for fast lookup
        ecg_peak_times = set()
        if hasattr(self.processorECG, "peak_unix_times"):
            for t in self.processorECG.peak_unix_times:
                ecg_peak_times.add(round(t, 3))
        ppg_peak_times = set()
        if hasattr(self.processorPPG, "peak_unix_times"):
            for t in self.processorPPG.peak_unix_times:
                ppg_peak_times.add(round(t, 3))

        # Slice arrays from found indices
        ecg_times_cut = ecg_times[idx_ecg:]
        ecg_data_cut = self.ecg_out_data[idx_ecg:]
        ppg_times_cut = ppg_times[idx_ppg:]
        ppg_data_cut = self.ppg_out_data[idx_ppg:]

        # Create peak arrays (0/1) for each sample
        ecg_peaks_arr = [1 if round(t, 3) in ecg_peak_times else 0 for t in ecg_times_cut]
        ppg_peaks_arr = [1 if round(t, 3) in ppg_peak_times else 0 for t in ppg_times_cut]

        # Save ECG with peaks
        df_ecg = pd.DataFrame({
            'time': ecg_times_cut,
            'ecg': ecg_data_cut,
            'peak': ecg_peaks_arr,
        })
        df_ecg.to_csv('ecg_data_aligned.csv', index=False)
        print("ECG data saved to ecg_data_aligned.csv")

        # Save PPG with peaks
        df_ppg = pd.DataFrame({
            'time': ppg_times_cut,
            'ppg': ppg_data_cut,
            'peak': ppg_peaks_arr,
        })
        df_ppg.to_csv('ppg_data_aligned.csv', index=False)
        print("PPG data saved to ppg_data_aligned.csv")

    def save_hrv_plots_png(self):
        """
        Save all HRV tab plots (EKG and PPG) as PNG files with 300 DPI.
        """
        for key, plot_obj in self.hrv_plots.items():
            # The plot.LivePlot class should have a save_png method
            filename = f"{key}.png"
            try:
                plot_obj.save_png(filename, dpi=300)
                print(f"Saved {filename} at 300 DPI")
            except Exception as e:
                print(f"Failed to save {filename}: {e}")

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