import queue
import random
import threading
from math import sin, pi, cos
from random import randint
import tkinter as tk

import numpy as np

from appgui import plot, data
from appgui.EcgDataFile import EcgDataFile
from appgui.PPGProgessor import PPGProcessor
from appgui.PpgData import PpgData
from appgui.control import ControlPanel
from appgui.ECGProcessor import ECGProcessor


class LiveCounterApp:
    def __init__(self, root):
        self.counter2 = 0

        self.root = root
        self.root.title("HR")

        self.counter = 0
        self.is_running = True
        self.processorECG = ECGProcessor(window_size=256)
        self.processorPPG = PPGProcessor()

        self.controls = ControlPanel(
            root,
            pause_callback=self.pause,
            resume_callback=self.resume,
            reset_callback=self.reset
        )
        self.controls.pack(pady=10)
        frame1 = tk.Frame(root)
        frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        frame2 = tk.Frame(root)
        frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.queueECG = queue.Queue()
        self.queuePPG = queue.Queue()

        self.stop_event_PPG = threading.Event()
        self.stop_event_ECG = threading.Event()

        self.plotECG = plot.LivePlot(frame1, "ECG", "Time [ms]", "Signal [normalized]")
        self.plotPPG = plot.LivePlot(frame2, "PPG", "Time [ms]", "Signal [normalized]", y_lim=(50,70))


        self.thread2 = PpgData(self.queuePPG, self.stop_event_PPG)

        self.thread1 = EcgDataFile(self.queueECG, self.stop_event_ECG)

        self.thread1.start()
        self.thread2.start()

        self.update_loop()

    def update_loop(self):
        if self.is_running:
            try:
                val1 = self.queueECG.get_nowait()
                self.plotECG.add_data(val1)
                self.counter += 1
                result = self.processorECG.add_sample(val1, self.counter)
                if result is not None:
                    self.plotECG.add_scatter_points(result.x_peaks, result.y_peaks)

            except queue.Empty:
                pass
            try:
                while True:
                    val2 = self.queuePPG.get_nowait()
                    self.counter2 += 1
                    result = self.processorPPG.add_sample(val2, self.counter2)
                    if result is not None:
                        print(val2)
                        d = result.filtered_signal
                        raw = result.raw_signal
                        for r in range(len(raw)):
                            self.plotPPG.add_data(raw[r], d[r])
                        self.plotPPG.add_scatter_points(result.peak_times, result.peak_values)
            except queue.Empty:
                pass

        self.root.after(10, self.update_loop)

    def pause(self):
        self.is_running = False

    def resume(self):
        self.is_running = True

    def reset(self):
        self.plotPPG.reset()
        self.plotECG.reset()

    def on_closing(self):
        self.stop_event_PPG.set()
        self.stop_event_ECG.set()
        self.thread1.join(timeout=1)
        self.thread2.join(timeout=1)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCounterApp(root)
    root.mainloop()
