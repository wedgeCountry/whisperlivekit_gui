import math
import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Callable

import numpy as np

from .theme import (
    C_BG, C_SURFACE, C_BORDER, C_TEXT, C_MUTED,
    C_ACCENT, C_ACCENT_H, F_SMALL, center_on_parent,
    apply_ttk_style,
)
from transcribe_app.config import CHANNELS, DTYPE, SAMPLE_RATE
from transcribe_app.settings import Settings


def _list_input_devices() -> list[tuple[int, str]]:
    """Return [(index, label), …] for all devices with at least one input channel."""
    import sounddevice as sd
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append((i, dev["name"]))
    return devices


class MicTestWindow:
    BAR_W   = 420
    BAR_H   = 32
    POLL_MS = 40
    CHUNK   = 1024

    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings   = settings
        self._on_save    = on_save
        self._stream     = None
        self._amplitude  = 0.0
        self._peak_x     = 0.0
        self._peak_decay = 0.0

        self._win = tk.Toplevel(parent)
        self._win.title("Microphone Test")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        apply_ttk_style(self._win)
        self._build_ui()
        center_on_parent(self._win, parent)
        self._start_stream(settings.input_device)
        self._animate()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Device selector ────────────────────────────────────────────────────
        sel_frame = tk.Frame(outer, bg=C_BG)
        sel_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            sel_frame, text="Input device",
            bg=C_BG, fg=C_MUTED, font=F_SMALL,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 8))

        try:
            self._devices = _list_input_devices()
        except Exception:
            self._devices = []

        # Build display labels; mark the system default
        labels = ["System default"] + [name for _, name in self._devices]
        indices = [None] + [idx for idx, _ in self._devices]  # parallel list

        # Find the label matching the current setting
        current_idx = self._settings.input_device
        try:
            sel = indices.index(current_idx)
        except ValueError:
            sel = 0

        self._dev_var = tk.StringVar(value=labels[sel])
        self._dev_indices = indices

        combo = ttk.Combobox(
            sel_frame,
            textvariable=self._dev_var,
            values=labels,
            state="readonly",
            width=36,
            font=("TkDefaultFont", 10),
        )
        combo.current(sel)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self._on_device_change)

        # ── Level label ────────────────────────────────────────────────────────
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

    def _on_device_change(self, _event=None) -> None:
        label = self._dev_var.get()
        labels = ["System default"] + [name for _, name in self._devices]
        sel = labels.index(label)
        device = self._dev_indices[sel]

        self._settings = replace(self._settings, input_device=device)
        self._on_save(self._settings)

        # Restart the stream with the new device
        self._stop_stream()
        self._amplitude  = 0.0
        self._peak_x     = 0.0
        self._peak_decay = 0.0
        self._start_stream(device)

    def _start_stream(self, device: int | None) -> None:
        import sounddevice as sd

        def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            self._amplitude = min(rms / 32768.0, 1.0)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=self.CHUNK, callback=_cb,
            device=device,
        )
        self._stream.start()

    def _stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
        self._stop_stream()
        self._win.destroy()
