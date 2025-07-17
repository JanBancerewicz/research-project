import threading

import time

class DataProducerThread(threading.Thread):
    def __init__(self, data_queue, stop_event, signal_func, name="Producer"):
        super().__init__()
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.signal_func = signal_func
        self.counter = 0
        self.name = name

    def run(self):
        while not self.stop_event.is_set():
            val = self.signal_func(self.counter)
            self.data_queue.put(val)
            self.counter += 1
            time.sleep(0.01)
