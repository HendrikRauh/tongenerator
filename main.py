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

        phase_offset = 2 * np.pi * phase_diff / 360
        left = np.sin(2 * np.pi * freq_left * t + self.phase_left)
        right = np.sin(2 * np.pi * freq_right * t + self.phase_right + phase_offset)

        if mute_left:
            left = np.zeros_like(left)
        if mute_right:
            right = np.zeros_like(right)

        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

        self.phase_left += 2 * np.pi * freq_left * frames / SAMPLE_RATE
        self.phase_left = self.phase_left % (2 * np.pi)
        self.phase_right += 2 * np.pi * freq_right * frames / SAMPLE_RATE
        self.phase_right = self.phase_right % (2 * np.pi)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interferenz-Experiment")
        self.resizable(True, True)

        self.configure(bg="#232629")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#232629", foreground="#f8f8f2", fieldbackground="#232629")
        style.configure("TLabel", background="#232629", foreground="#f8f8f2")
        style.configure("TButton", background="#44475a", foreground="#f8f8f2")
        style.configure("TCheckbutton", background="#232629", foreground="#f8f8f2")
        style.configure("Horizontal.TScale", background="#232629")
        style.configure("Vertical.TScale", background="#232629")
        style.map("TButton", background=[("active", "#6272a4")])
        style.configure("Borderless.TEntry", relief="flat", borderwidth=0, font=("Segoe UI", 12), foreground="#f8f8f2", fieldbackground="#232629", background="#232629")

        self.generator = ToneGenerator()

        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        faders_frame = ttk.Frame(main_frame)
        faders_frame.pack(side="top", fill="both", expand=True)

        left_frame = ttk.Frame(faders_frame)
        left_frame.pack(side="left", expand=True, fill="both", padx=10)
        ttk.Label(left_frame, text="Links (Hz)").pack(pady=(0, 8))
        self.freq_left = tk.IntVar(value=440)
        self.slider_left = ttk.Scale(left_frame, from_=2000, to=100, variable=self.freq_left, orient="vertical", command=self.update_left)
        self.slider_left.pack(expand=True, fill="y")
        self.entry_left = ttk.Entry(left_frame, width=6, justify="center")
        self.entry_left.insert(0, "440")
        self.entry_left.pack(pady=(8, 4))
        self.entry_left.bind("<Return>", self.set_left_from_entry)
        self.entry_left.bind("<FocusOut>", self.set_left_from_entry)
        self.entry_left.configure(style="Borderless.TEntry")
        self.mute_left = tk.BooleanVar(value=False)
        self.btn_mute_left = ttk.Button(
            left_frame,
            text="🔈",
            width=3,
            command=self.toggle_mute_left
        )
        self.btn_mute_left.pack(pady=(8, 0))

        phase_frame = ttk.Frame(faders_frame)
        phase_frame.pack(side="left", expand=True, fill="both", padx=10)
        ttk.Label(phase_frame, text="Phase (°)").pack(pady=(0, 8))
        self.phase_diff = tk.IntVar(value=0)
        self.slider_phase = ttk.Scale(phase_frame, from_=360, to=0, variable=self.phase_diff, orient="vertical", command=self.update_phase)
        self.slider_phase.pack(expand=True, fill="y")
        self.entry_phase = ttk.Entry(phase_frame, width=6, justify="center")
        self.entry_phase.insert(0, "0")
        self.entry_phase.pack(pady=(8, 4))
        self.entry_phase.bind("<Return>", self.set_phase_from_entry)
        self.entry_phase.bind("<FocusOut>", self.set_phase_from_entry)
        self.entry_phase.configure(style="Borderless.TEntry")

        right_frame = ttk.Frame(faders_frame)
        right_frame.pack(side="left", expand=True, fill="both", padx=10)
        ttk.Label(right_frame, text="Rechts (Hz)").pack(pady=(0, 8))
        self.freq_right = tk.IntVar(value=440)
        self.slider_right = ttk.Scale(right_frame, from_=2000, to=100, variable=self.freq_right, orient="vertical", command=self.update_right)
        self.slider_right.pack(expand=True, fill="y")
        self.entry_right = ttk.Entry(right_frame, width=6, justify="center")
        self.entry_right.insert(0, "440")
        self.entry_right.pack(pady=(8, 4))
        self.entry_right.bind("<Return>", self.set_right_from_entry)
        self.entry_right.bind("<FocusOut>", self.set_right_from_entry)
        self.entry_right.configure(style="Borderless.TEntry")
        self.mute_right = tk.BooleanVar(value=False)
        self.btn_mute_right = ttk.Button(
            right_frame,
            text="🔈",
            width=3,
            command=self.toggle_mute_right
        )
        self.btn_mute_right.pack(pady=(8, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side="bottom", fill="x", pady=(20, 0))
        self.btn_start = ttk.Button(button_frame, text="Start", command=self.start)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.btn_stop = ttk.Button(button_frame, text="Stop", command=self.stop)
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=(10, 0))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_left(self, event=None):
        freq = self.freq_left.get()
        self.entry_left.delete(0, tk.END)
        self.entry_left.insert(0, str(int(freq)))
        self.generator.set_freq_left(freq)

    def set_left_from_entry(self, event=None):
        try:
            freq = int(self.entry_left.get())
            freq = max(100, min(2000, freq))
            self.freq_left.set(freq)
            self.update_left()
        except ValueError:
            pass

    def update_right(self, event=None):
        freq = self.freq_right.get()
        self.entry_right.delete(0, tk.END)
        self.entry_right.insert(0, str(int(freq)))
        self.generator.set_freq_right(freq)

    def set_right_from_entry(self, event=None):
        try:
            freq = int(self.entry_right.get())
            freq = max(100, min(2000, freq))
            self.freq_right.set(freq)
            self.update_right()
        except ValueError:
            pass

    def update_phase(self, event=None):
        diff = self.phase_diff.get()
        self.entry_phase.delete(0, tk.END)
        self.entry_phase.insert(0, str(int(diff)))
        self.generator.set_phase_diff(diff)

    def set_phase_from_entry(self, event=None):
        try:
            diff = int(self.entry_phase.get())
            diff = max(0, min(360, diff))
            self.phase_diff.set(diff)
            self.update_phase()
        except ValueError:
            pass

    def start(self):
        self.generator.set_freq_left(self.freq_left.get())
        self.generator.set_freq_right(self.freq_right.get())
        self.generator.set_phase_diff(self.phase_diff.get())
        self.generator.start()

    def stop(self):
        self.generator.stop()

    def toggle_mute_left(self):
        current = self.mute_left.get()
        self.mute_left.set(not current)
        self.generator.set_mute_left(not current)
        self.btn_mute_left.config(text="🔇" if not current else "🔈")

    def toggle_mute_right(self):
        current = self.mute_right.get()
        self.mute_right.set(not current)
        self.generator.set_mute_right(not current)
        self.btn_mute_right.config(text="🔇" if not current else "🔈")

    def on_close(self):
        self.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()