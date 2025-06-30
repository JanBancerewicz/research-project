import pandas as pd

from appgui.data import DataProducerThread


class EcgDataFile(DataProducerThread):
    def __init__(self, data_queue, stop_event, name="EcgDataFile"):
        DataProducerThread.__init__(self, data_queue, stop_event, self.signal_func, name,)
        self.file = pd.read_csv("data/night_R2.csv")

    def signal_func(self, counter):
        return self.file["ecg"][counter]