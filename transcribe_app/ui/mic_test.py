import math
import tkinter as tk

import numpy as np

from .theme import (
    C_BG, C_SURFACE, C_BORDER, C_TEXT, C_MUTED,
    C_ACCENT, C_ACCENT_H, F_SMALL, center_on_parent,
)
from transcribe_app.config import CHANNELS, DTYPE, SAMPLE_RATE


class MicTestWindow:
    BAR_W   = 420
    BAR_H   = 32
    POLL_MS = 40
    CHUNK   = 1024

    def __init__(self, parent: tk.Widget) -> None:
        self._stream     = None
        self._amplitude  = 0.0
        self._peak_x     = 0.0
        self._peak_decay = 0.0

        self._win = tk.Toplevel(parent)
        self._win.title("Microphone Test")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        center_on_parent(self._win, parent)
        self._start_stream()
        self._animate()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            outer, text="Microphone input level",
            bg=C_BG, fg=C_TEXT, font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))

        bar_border = tk.Frame(outer, bg=C_BORDER)
        bar_border.pack(fill=tk.X, pady=(0, 8))

        self._canvas = tk.Canvas(
            bar_border,
            width=self.BAR_W, height=self.BAR_H,
            bg=C_SURFACE, highlightthickness=0,
        )
        self._canvas.pack(padx=1, pady=1)
        self._canvas.create_rectangle(0, 0, 0, self.BAR_H, fill=C_ACCENT,   outline="", tags="bar")
        self._canvas.create_rectangle(0, 0, 0, self.BAR_H, fill=C_ACCENT_H, outline="", tags="peak")

        self._db_var = tk.StringVar(value="–∞ dBFS")
        tk.Label(
            outer, textvariable=self._db_var,
            bg=C_BG, fg=C_TEXT, font=("TkFixedFont", 12),
            anchor="center",
        ).pack(fill=tk.X, pady=(0, 8))

        try:
            import sounddevice as sd
            dev  = sd.query_devices(kind="input")
            info = f"{dev['name']}  ·  {int(dev['default_samplerate'])} Hz"
        except Exception:
            info = "Default input device"
        tk.Label(
            outer, text=info,
            bg=C_BG, fg=C_MUTED, font=F_SMALL,
            anchor="center", wraplength=self.BAR_W,
        ).pack(fill=tk.X)

    def _start_stream(self) -> None:
        import sounddevice as sd

        def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            self._amplitude = min(rms / 32768.0, 1.0)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=self.CHUNK, callback=_cb,
        )
        self._stream.start()

    def _animate(self) -> None:
        if not self._win.winfo_exists():
            return

        amp = self._amplitude
        if amp >= self._peak_x:
            self._peak_x     = amp
            self._peak_decay = 0.0
        else:
            self._peak_decay += self.POLL_MS / 1000.0
            self._peak_x = max(self._peak_x - self._peak_decay * 0.4, amp)

        bar_px  = int(amp          * self.BAR_W)
        peak_px = int(self._peak_x * self.BAR_W)

        if amp < 0.5:
            r, g = int(amp * 2 * 210), 180
        else:
            r, g = 210, int((1.0 - (amp - 0.5) * 2) * 180)
        colour = f"#{r:02x}{g:02x}50"

        self._canvas.itemconfig("bar",  fill=colour)
        self._canvas.coords("bar",  0, 2, max(bar_px, 0),  self.BAR_H - 2)
        self._canvas.coords("peak", peak_px - 2, 0, peak_px + 2, self.BAR_H)
        self._db_var.set(f"{20 * math.log10(amp):+.1f} dBFS" if amp > 0 else "–∞ dBFS")
        self._win.after(self.POLL_MS, self._animate)

    def _on_close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._win.destroy()
