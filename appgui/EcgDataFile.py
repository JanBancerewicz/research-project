import pandas as pd
import os
from appgui.data import DataProducerThread


class EcgDataFile(DataProducerThread):
    def __init__(self, data_queue, stop_event, name="EcgDataFile"):
        DataProducerThread.__init__(self, data_queue, stop_event, self.signal_func, name, )

        # --- PATH FIX ---
        # Current file is in appgui/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to research-project/
        project_root = os.path.dirname(current_dir)
        # Build path to data/night_R2.csv
        file_path = os.path.join(project_root, "data", "night_R2.csv")

        if not os.path.exists(file_path):
            print(f"CRITICAL ERROR: ECG file not found at {file_path}")
            # Fallback or error handling? Usually just crash with a better message or let pandas crash

        self.file = pd.read_csv(file_path)

    def signal_func(self, counter):
        return counter, self.file["ecg"][counter]