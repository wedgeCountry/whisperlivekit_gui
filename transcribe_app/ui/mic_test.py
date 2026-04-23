import math
import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Callable

import numpy as np

from .theme import (
    C_ACCENT,
    C_ACCENT_H,
    C_BG,
    C_BORDER,
    C_INPUT,
    C_MUTED,
    C_STATUS_BG,
    C_SURFACE,
    C_TEXT,
    F_SMALL,
    apply_ttk_style,
    center_on_parent,
    make_card,
    style_scale_widget,
)
from transcribe_app.config import CHANNELS, DTYPE, IS_WINDOWS, SAMPLE_RATE
from transcribe_app.i18n import t
from transcribe_app.settings import Settings

_GAIN_MIN = 0.1
_GAIN_MAX = 5.0


def _list_input_devices() -> list[tuple[int, str]]:
    """Return [(index, label), ...] for all devices with at least one input channel."""
    import sounddevice as sd

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append((i, dev["name"]))
    return devices


def _device_uses_wasapi(sd, device: int | None) -> bool:
    try:
        resolved = sd.default.device[0] if device is None else device
        info = sd.query_devices(resolved, "input")
        hostapi_index = info.get("hostapi")
        hostapi_info = sd.query_hostapis(hostapi_index)
        return "wasapi" in str(hostapi_info.get("name", "")).lower()
    except Exception:
        return False


class MicTestWindow:
    BAR_W = 420
    BAR_H = 32
    POLL_MS = 40
    CHUNK = 1024

    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save = on_save
        self._stream = None
        self._amplitude = 0.0
        self._peak_x = 0.0
        self._peak_decay = 0.0

        self._win = tk.Toplevel(parent)
        self._win.title(t("dlg.mic.title"))
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        apply_ttk_style(self._win)
        self._build_ui()
        center_on_parent(self._win, parent)
        self._start_stream(settings.input_device)
        self._animate()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        header = tk.Frame(outer, bg=C_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            header,
            text=t("dlg.mic.title"),
            bg=C_BG,
            fg=C_TEXT,
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=t("dlg.mic.input_device"),
            bg=C_BG,
            fg=C_MUTED,
            font=F_SMALL,
        ).pack(anchor="w", pady=(3, 0))

        try:
            self._devices = _list_input_devices()
        except Exception:
            self._devices = []

        labels = [t("dlg.mic.default_device")] + [name for _, name in self._devices]
        indices = [None] + [idx for idx, _ in self._devices]

        current_idx = self._settings.input_device
        try:
            sel = indices.index(current_idx)
        except ValueError:
            sel = 0

        device_card = make_card(outer, padx=14, pady=14)
        device_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        device_card.columnconfigure(0, weight=1)

        self._dev_var = tk.StringVar(value=labels[sel])
        self._dev_indices = indices

        tk.Label(
            device_card,
            text=t("dlg.mic.input_device"),
            bg=C_SURFACE,
            fg=C_MUTED,
            font=F_SMALL,
        ).grid(row=0, column=0, sticky="w")

        combo = ttk.Combobox(
            device_card,
            textvariable=self._dev_var,
            values=labels,
            state="readonly",
            width=40,
            font=("TkDefaultFont", 10),
            style="Modern.TCombobox",
        )
        combo.current(sel)
        combo.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        combo.bind("<<ComboboxSelected>>", self._on_device_change)

        level_card = make_card(outer, padx=14, pady=14)
        level_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        level_card.columnconfigure(0, weight=1)

        tk.Label(
            level_card,
            text=t("dlg.mic.level"),
            bg=C_SURFACE,
            fg=C_TEXT,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            level_card,
            text=t("dlg.mic.sensitivity"),
            bg=C_SURFACE,
            fg=C_MUTED,
            font=F_SMALL,
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        meter_shell = make_card(level_card, bg=C_INPUT, border=C_BORDER, padx=1, pady=1)
        meter_shell.grid(row=2, column=0, sticky="ew")
        self._canvas = tk.Canvas(
            meter_shell,
            width=self.BAR_W,
            height=self.BAR_H,
            bg=C_INPUT,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill=tk.X)
        self._canvas.create_rectangle(0, 0, 0, self.BAR_H, fill=C_ACCENT, outline="", tags="bar")
        self._canvas.create_rectangle(0, 0, 0, self.BAR_H, fill=C_ACCENT_H, outline="", tags="peak")

        self._db_var = tk.StringVar(value="-inf dBFS")
        tk.Label(
            level_card,
            textvariable=self._db_var,
            bg=C_SURFACE,
            fg=C_TEXT,
            font=("TkFixedFont", 12),
            anchor="center",
        ).grid(row=3, column=0, sticky="ew", pady=(10, 0))

        gain_card = make_card(outer, padx=14, pady=14)
        gain_card.grid(row=3, column=0, sticky="ew")
        gain_card.columnconfigure(0, weight=1)

        tk.Label(
            gain_card,
            text=t("dlg.mic.sensitivity"),
            bg=C_SURFACE,
            fg=C_TEXT,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            gain_card,
            text=t("dlg.mic.level"),
            bg=C_SURFACE,
            fg=C_MUTED,
            font=F_SMALL,
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        slider_row = tk.Frame(gain_card, bg=C_SURFACE)
        slider_row.grid(row=2, column=0, sticky="ew")
        slider_row.columnconfigure(0, weight=1)

        self._gain_var = tk.DoubleVar(value=self._settings.mic_gain)
        self._gain_label_var = tk.StringVar(value=self._gain_db_str(self._settings.mic_gain))

        scale = tk.Scale(
            slider_row,
            variable=self._gain_var,
            from_=_GAIN_MIN,
            to=_GAIN_MAX,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_gain_change,
        )
        style_scale_widget(scale, trough=C_STATUS_BG)
        scale.grid(row=0, column=0, sticky="ew")

        tk.Label(
            slider_row,
            textvariable=self._gain_label_var,
            bg=C_SURFACE,
            fg=C_TEXT,
            font=("TkFixedFont", 11),
            width=10,
            anchor="e",
        ).grid(row=0, column=1, padx=(10, 0))

    @staticmethod
    def _gain_db_str(gain: float) -> str:
        db = 20 * math.log10(max(gain, 1e-9))
        return f"{db:+.1f} dB"

    def _on_gain_change(self, _value=None) -> None:
        gain = round(self._gain_var.get(), 5)
        self._gain_label_var.set(self._gain_db_str(gain))
        self._settings = replace(self._settings, mic_gain=gain)
        self._on_save(self._settings)

    def _on_device_change(self, _event=None) -> None:
        label = self._dev_var.get()
        labels = [t("dlg.mic.default_device")] + [name for _, name in self._devices]
        sel = labels.index(label)
        device = self._dev_indices[sel]

        self._settings = replace(self._settings, input_device=device)
        self._on_save(self._settings)

        self._stop_stream()
        self._amplitude = 0.0
        self._peak_x = 0.0
        self._peak_decay = 0.0
        self._start_stream(device)

    def _start_stream(self, device: int | None) -> None:
        import sounddevice as sd

        def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            self._amplitude = min(rms / 32768.0 * self._settings.mic_gain, 1.0)

        kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.CHUNK,
            callback=_cb,
            device=device,
        )
        if IS_WINDOWS and _device_uses_wasapi(sd, device):
            try:
                self._stream = sd.InputStream(**kwargs, extra_settings=sd.WasapiSettings(exclusive=True))
                self._stream.start()
                return
            except Exception:
                pass
        self._stream = sd.InputStream(**kwargs)
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
            self._peak_x = amp
            self._peak_decay = 0.0
        else:
            self._peak_decay += self.POLL_MS / 1000.0
            self._peak_x = max(self._peak_x - self._peak_decay * 0.4, amp)

        bar_px = int(amp * self.BAR_W)
        peak_px = int(self._peak_x * self.BAR_W)

        if amp < 0.5:
            r, g = int(amp * 2 * 210), 180
        else:
            r, g = 210, int((1.0 - (amp - 0.5) * 2) * 180)
        colour = f"#{r:02x}{g:02x}50"

        self._canvas.itemconfig("bar", fill=colour)
        self._canvas.coords("bar", 0, 2, max(bar_px, 0), self.BAR_H - 2)
        self._canvas.coords("peak", peak_px - 2, 0, peak_px + 2, self.BAR_H)
        self._db_var.set(f"{20 * math.log10(amp):+.1f} dBFS" if amp > 0 else "-inf dBFS")
        self._win.after(self.POLL_MS, self._animate)

    def _on_close(self) -> None:
        self._stop_stream()
        self._win.destroy()
