import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100

class ToneGenerator:
    def __init__(self):
        self.freq_left = 440
        self.freq_right = 440
        self.phase_diff = 0
        self.running = False
        self.stream = None
        self.lock = threading.Lock()
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.mute_left = False
        self.mute_right = False

    def start(self):
        if not self.running:
            self.running = True
            self.phase_left = 0.0
            self.phase_right = 0.0
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype='float32',
                callback=self.callback
            )
            self.stream.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_freq_left(self, freq):
        with self.lock:
            self.freq_left = freq

    def set_freq_right(self, freq):
        with self.lock:
            self.freq_right = freq

    def set_phase_diff(self, diff):
        with self.lock:
            self.phase_diff = diff

    def set_mute_left(self, mute):
        with self.lock:
            self.mute_left = mute

    def set_mute_right(self, mute):
        with self.lock:
            self.mute_right = mute

    def callback(self, outdata, frames, time, status):
        t = np.arange(frames) / SAMPLE_RATE
        with self.lock:
            freq_left = self.freq_left
            freq_right = self.freq_right
            phase_diff = self.phase_diff
            mute_left = self.mute_left
            mute_right = self.mute_right

        # Phasen in Radiant
        phase_offset = 2 * np.pi * phase_diff / 360

        # Fortlaufende Phase berechnen
        left = np.sin(2 * np.pi * freq_left * t + self.phase_left)
        right = np.sin(2 * np.pi * freq_right * t + self.phase_right + phase_offset)

        if mute_left:
            left = np.zeros_like(left)
        if mute_right:
            right = np.zeros_like(right)

        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

        # Phasen für nächsten Callback merken
        self.phase_left += 2 * np.pi * freq_left * frames / SAMPLE_RATE
        self.phase_left = self.phase_left % (2 * np.pi)
        self.phase_right += 2 * np.pi * freq_right * frames / SAMPLE_RATE
        self.phase_right = self.phase_right % (2 * np.pi)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interferenz-Experiment")
        self.geometry("400x320")
        self.resizable(False, False)

        self.generator = ToneGenerator()

        # Frequenz links
        ttk.Label(self, text="Frequenz links (Hz)").pack(pady=(10,0))
        self.freq_left = tk.IntVar(value=440)
        self.slider_left = ttk.Scale(self, from_=100, to=2000, variable=self.freq_left, orient="horizontal", command=self.update_left)
        self.slider_left.pack(fill="x", padx=20)
        self.label_left = ttk.Label(self, text="440 Hz")
        self.label_left.pack()

        # Frequenz rechts
        ttk.Label(self, text="Frequenz rechts (Hz)").pack(pady=(10,0))
        self.freq_right = tk.IntVar(value=440)
        self.slider_right = ttk.Scale(self, from_=100, to=2000, variable=self.freq_right, orient="horizontal", command=self.update_right)
        self.slider_right.pack(fill="x", padx=20)
        self.label_right = ttk.Label(self, text="440 Hz")
        self.label_right.pack()

        # Gangunterschied
        ttk.Label(self, text="Gangunterschied (°)").pack(pady=(10,0))
        self.phase_diff = tk.IntVar(value=0)
        self.slider_phase = ttk.Scale(self, from_=0, to=360, variable=self.phase_diff, orient="horizontal", command=self.update_phase)
        self.slider_phase.pack(fill="x", padx=20)
        self.label_phase = ttk.Label(self, text="0°")
        self.label_phase.pack()

        # Mute Checkbuttons
        mute_frame = ttk.Frame(self)
        mute_frame.pack(fill="x", padx=20, pady=(10,0))
        self.mute_left = tk.BooleanVar(value=False)
        self.mute_right = tk.BooleanVar(value=False)
        self.chk_mute_left = ttk.Checkbutton(mute_frame, text="Links stumm", variable=self.mute_left, command=self.update_mute_left)
        self.chk_mute_left.pack(side="left", expand=True, fill="x")
        self.chk_mute_right = ttk.Checkbutton(mute_frame, text="Rechts stumm", variable=self.mute_right, command=self.update_mute_right)
        self.chk_mute_right.pack(side="right", expand=True, fill="x")

        # Start/Stop Buttons in eigenen Frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=20, pady=20)

        self.btn_start = ttk.Button(button_frame, text="Start", command=self.start)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.btn_stop = ttk.Button(button_frame, text="Stop", command=self.stop)
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=(10, 0))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_left(self, event=None):
        freq = self.freq_left.get()
        self.label_left.config(text=f"{int(freq)} Hz")
        self.generator.set_freq_left(freq)

    def update_right(self, event=None):
        freq = self.freq_right.get()
        self.label_right.config(text=f"{int(freq)} Hz")
        self.generator.set_freq_right(freq)

    def update_phase(self, event=None):
        diff = self.phase_diff.get()
        self.label_phase.config(text=f"{int(diff)}°")
        self.generator.set_phase_diff(diff)

    def start(self):
        self.generator.set_freq_left(self.freq_left.get())
        self.generator.set_freq_right(self.freq_right.get())
        self.generator.set_phase_diff(self.phase_diff.get())
        self.generator.start()

    def stop(self):
        self.generator.stop()

    def update_mute_left(self):
        self.generator.set_mute_left(self.mute_left.get())

    def update_mute_right(self):
        self.generator.set_mute_right(self.mute_right.get())

    def on_close(self):
        self.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()