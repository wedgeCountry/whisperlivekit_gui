import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from typing import Callable

from ..theme import C_BG, C_SURFACE, C_TEXT, C_BORDER, C_MUTED, F_SMALL, F_LABEL, make_btn, center_on_parent
from transcribe_app.config import GPU, LANGUAGE_OPTS, DEFAULT_PROMPTS, get_model_size
from transcribe_app.engine_protocol import ENGINE_LABELS, ENGINE_TYPES
from transcribe_app.i18n import UI_LANGUAGES, t
from transcribe_app.settings import Settings


class SettingsDialog:
    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save  = on_save

        self._win = tk.Toplevel(parent)
        self._win.title(t("dlg.settings.title"))
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.grab_set()
        self._build_ui()
        center_on_parent(self._win, parent)

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        # Row 0 — language
        tk.Label(outer, text=t("dlg.settings.language"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._lang_var = tk.StringVar(value=self._settings.language)
        ttk.Combobox(
            outer,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly",
            width=14,
            font=F_LABEL,
        ).grid(row=0, column=1, sticky="w", pady=(0, 10))

        # Row 1 — model speed
        tk.Label(outer, text=t("dlg.settings.model"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        speed_frame = tk.Frame(outer, bg=C_BG)
        speed_frame.grid(row=1, column=1, sticky="w", pady=(0, 10))

        self._speed_var = tk.StringVar(value=t(f"speed.{self._settings.model_speed}"))
        self._speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self._speed_var,
            values=[t("speed.fast"), t("speed.normal")],
            state="readonly",
            width=8,
            font=F_LABEL,
        )
        self._speed_combo.pack(side=tk.LEFT)

        self._speed_hint = tk.Label(
            speed_frame, bg=C_BG, fg=C_MUTED, font=F_SMALL, anchor="w",
        )
        self._speed_hint.pack(side=tk.LEFT, padx=(10, 0))
        # _update_speed_hint() called after _device_var is created (row 2)

        self._speed_var.trace_add("write", lambda *_: self._update_speed_hint())

        # Row 2 — compute device
        tk.Label(outer, text=t("dlg.settings.device"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._device_var = tk.StringVar(value=self._settings.compute_device)
        self._device_combo = ttk.Combobox(
            outer,
            textvariable=self._device_var,
            values=["cuda", "cpu"] if GPU else ["cpu"],
            state="readonly" if GPU else "disabled",
            width=6,
            font=F_LABEL,
        )
        self._device_combo.grid(row=2, column=1, sticky="w", pady=(0, 10))
        if not GPU:
            self._device_var.set("cpu")
        self._device_var.trace_add("write", lambda *_: self._update_speed_hint())
        self._update_speed_hint()  # now safe: _device_var exists

        # Row 3 — interface language
        tk.Label(outer, text=t("dlg.settings.ui_language"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._ui_lang_var = tk.StringVar(value=UI_LANGUAGES[self._settings.ui_language])
        ttk.Combobox(
            outer,
            textvariable=self._ui_lang_var,
            values=list(UI_LANGUAGES.values()),
            state="readonly",
            width=14,
            font=F_LABEL,
        ).grid(row=3, column=1, sticky="w", pady=(0, 10))

        # Row 4 — ASR post-processing
        tk.Label(outer, text=t("dlg.settings.asr"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=4, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._asr_var = tk.BooleanVar(value=self._settings.asr_postprocess)
        ttk.Checkbutton(outer, variable=self._asr_var).grid(row=4, column=1, sticky="w", pady=(0, 10))

        # Row 5 — engine backend
        tk.Label(outer, text=t("dlg.settings.engine"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=5, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._engine_var = tk.StringVar(value=ENGINE_LABELS[self._settings.engine_type])
        ttk.Combobox(
            outer,
            textvariable=self._engine_var,
            values=[ENGINE_LABELS[k] for k in ENGINE_TYPES],
            state="readonly",
            width=24,
            font=F_LABEL,
        ).grid(row=5, column=1, sticky="w", pady=(0, 10))

        # Row 6 — VAD silence gap
        tk.Label(outer, text=t("dlg.settings.vad_silence"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=6, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._silence_var = tk.StringVar(value=f"{self._settings.vad_silence_gap:.1f}")
        self._silence_spin = ttk.Spinbox(
            outer,
            textvariable=self._silence_var,
            from_=0.1, to=10.0, increment=0.1,
            width=5, font=F_LABEL, format="%.1f",
        )
        self._silence_spin.grid(row=6, column=1, sticky="w", pady=(0, 10))
        self._engine_var.trace_add("write", lambda *_: self._update_silence_state())
        self._update_silence_state()

        # Row 7 — style prompt
        tk.Label(outer, text=t("dlg.settings.prompt"), bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=7, column=0, sticky="nw", padx=(0, 10), pady=(0, 12)
        )
        border = tk.Frame(outer, bg=C_BORDER)
        border.grid(row=7, column=1, sticky="ew", pady=(0, 12))

        self._prompt_text = tk.Text(
            border,
            wrap=tk.WORD, height=4, font=F_SMALL,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, bd=0, padx=8, pady=6,
        )
        self._prompt_text.pack(padx=1, pady=1)
        self._prompt_text.insert(tk.END, self._settings.prompts[self._settings.language])

        self._lang_var.trace_add("write", self._on_lang_change)

        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=8, column=0, columnspan=2, sticky="e")
        make_btn(btn_row, t("dlg.settings.reset"), self._reset).pack(side=tk.LEFT, padx=(0, 8))
        make_btn(btn_row, t("dlg.settings.save"), self._save, primary=True).pack(side=tk.LEFT)

    def _update_silence_state(self) -> None:
        engine_key = next((k for k, v in ENGINE_LABELS.items() if v == self._engine_var.get()), "whisperlive")
        state = "normal" if engine_key == "faster_whisper" else "disabled"
        self._silence_spin.config(state=state)

    @staticmethod
    def _speed_key(label: str) -> str:
        """Reverse-map a translated speed label to its internal key."""
        return next((s for s in ("fast", "normal") if t(f"speed.{s}") == label), "fast")

    def _update_speed_hint(self) -> None:
        lang     = self._lang_var.get()
        speed    = self._speed_key(self._speed_var.get())
        use_gpu  = GPU and self._device_var.get() == "cuda"
        if lang and speed:
            model = get_model_size(lang, speed, use_gpu)
            self._speed_hint.config(text=f"({model})")

    def _on_lang_change(self, *_) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, self._settings.prompts[lang])
        self._update_speed_hint()

    def _reset(self) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, DEFAULT_PROMPTS[lang])

    def _save(self) -> None:
        lang           = self._lang_var.get()
        speed          = self._speed_key(self._speed_var.get())
        ui_lang        = next((k for k, v in UI_LANGUAGES.items() if v == self._ui_lang_var.get()), "en")
        compute_device = self._device_var.get()
        engine_type    = next((k for k, v in ENGINE_LABELS.items() if v == self._engine_var.get()), "whisperlive")
        new_prompts    = {**self._settings.prompts, lang: self._prompt_text.get("1.0", tk.END).strip()}
        try:
            silence_gap = round(max(0.1, min(10.0, float(self._silence_var.get()))), 1)
        except ValueError:
            silence_gap = 0.8
        self._on_save(replace(
            self._settings,
            language=lang, prompts=new_prompts, model_speed=speed,
            ui_language=ui_lang, compute_device=compute_device,
            asr_postprocess=self._asr_var.get(),
            engine_type=engine_type,
            vad_silence_gap=silence_gap,
        ))
        self._win.destroy()
