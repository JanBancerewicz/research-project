import tkinter as tk

class ControlPanel(tk.Frame):
    def __init__(self, parent, pause_callback, resume_callback, reset_callback):
        super().__init__(parent)
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self.reset_callback = reset_callback

        btn_pause = tk.Button(self, text="Pause", command=self.pause_callback)
        btn_pause.pack(side=tk.LEFT, padx=5)

        btn_resume = tk.Button(self, text="Resume", command=self.resume_callback)
        btn_resume.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(self, text="Reset", command=self.reset_callback)
        btn_reset.pack(side=tk.LEFT, padx=5)
