import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from typing import Callable

from ..theme import (
    C_ACCENT,
    C_BG,
    C_BORDER,
    C_MUTED,
    C_SURFACE,
    C_TEXT,
    F_LABEL,
    F_SMALL,
    center_on_parent,
    make_btn,
)
from transcribe_app.config import DEFAULT_PROMPTS, GPU, LANGUAGE_OPTS
from transcribe_app.engine_protocol import ENGINE_LABELS, ENGINE_TYPES
from transcribe_app.i18n import UI_LANGUAGES, t
from transcribe_app.settings import Settings


class SettingsDialog:
    _META_FONT = ("TkDefaultFont", 8)

    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save = on_save
        self._parent = parent

        self._win = tk.Toplevel(parent)
        self._win.title(t("dlg.settings.title"))
        self._win.resizable(True, True)
        self._win.minsize(520, 360)
        self._win.configure(bg=C_BG)
        self._win.grab_set()
        self._build_ui()
        self._apply_initial_size()
        center_on_parent(self._win, parent)

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=14, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = tk.Frame(outer, bg=C_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            header,
            text=t("dlg.settings.title"),
            bg=C_BG,
            fg=C_TEXT,
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=t("dlg.settings.general_hint"),
            bg=C_BG,
            fg=C_MUTED,
            font=self._META_FONT,
        ).pack(anchor="w", pady=(3, 0))
        tk.Frame(header, bg=C_BORDER, height=1).pack(fill=tk.X, pady=(10, 0))

        form_shell = tk.Frame(
            outer,
            bg=C_SURFACE,
            highlightbackground=C_BORDER,
            highlightthickness=1,
            bd=0,
        )
        form_shell.grid(row=1, column=0, sticky="nsew")
        form_shell.columnconfigure(0, weight=1)
        form_shell.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            form_shell,
            bg=C_SURFACE,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(
            form_shell,
            orient=tk.VERTICAL,
            command=canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        form = tk.Frame(
            canvas,
            bg=C_SURFACE,
            padx=14,
            pady=14,
        )
        form.columnconfigure(0, minsize=132)
        form.columnconfigure(1, weight=1)
        form_window = canvas.create_window((0, 0), window=form, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(_event) -> None:
            canvas.itemconfigure(form_window, width=_event.width)

        def _on_mousewheel(event) -> str:
            delta = event.delta
            if delta:
                canvas.yview_scroll(int(-delta / 120), "units")
            return "break"

        form.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        form.bind("<MouseWheel>", _on_mousewheel)

        self._lang_var = tk.StringVar(value=self._settings.language)
        self._add_row(
            form,
            0,
            t("dlg.settings.language"),
            self._make_combo(form, self._lang_var, list(LANGUAGE_OPTS.keys()), width=18),
        )

        self._speed_var = tk.StringVar(value=t(f"speed.{self._settings.model_speed}"))
        self._speed_combo = ttk.Combobox(
            form,
            textvariable=self._speed_var,
            values=[t("speed.fast"), t("speed.normal")],
            state="readonly",
            width=10,
            font=F_LABEL,
            style="Modern.TCombobox",
        )
        self._add_row(form, 1, t("dlg.settings.model"), self._speed_combo)

        self._device_var = tk.StringVar(value=self._settings.compute_device)
        self._device_combo = ttk.Combobox(
            form,
            textvariable=self._device_var,
            values=["cuda", "cpu"] if GPU else ["cpu"],
            state="readonly" if GPU else "disabled",
            width=10,
            font=F_LABEL,
            style="Modern.TCombobox",
        )
        self._add_row(form, 2, t("dlg.settings.device"), self._device_combo)
        if not GPU:
            self._device_var.set("cpu")

        self._ui_lang_var = tk.StringVar(value=UI_LANGUAGES[self._settings.ui_language])
        self._add_row(
            form,
            3,
            t("dlg.settings.ui_language"),
            self._make_combo(form, self._ui_lang_var, list(UI_LANGUAGES.values()), width=18),
        )

        self._asr_var = tk.BooleanVar(value=self._settings.asr_postprocess)
        self._cleanup_var = tk.BooleanVar(value=self._settings.cleanup_recordings)
        toggles_wrap = tk.Frame(form, bg=C_SURFACE)
        self._add_toggle_pair(
            toggles_wrap,
            [
                (t("dlg.settings.asr"), self._asr_var),
                (t("dlg.settings.cleanup_recordings"), self._cleanup_var),
            ],
        )
        self._add_row(form, 4, "", toggles_wrap, sticky="w")

        self._engine_var = tk.StringVar(value=ENGINE_LABELS[self._settings.engine_type])
        self._add_row(
            form,
            5,
            t("dlg.settings.engine"),
            self._make_combo(form, self._engine_var, [ENGINE_LABELS[k] for k in ENGINE_TYPES], width=28),
        )

        self._silence_var = tk.StringVar(value=f"{self._settings.vad_silence_gap:.1f}")
        self._silence_spin = ttk.Spinbox(
            form,
            textvariable=self._silence_var,
            from_=0.1,
            to=10.0,
            increment=0.1,
            width=6,
            font=F_LABEL,
            format="%.1f",
            style="Modern.TSpinbox",
        )
        self._add_row(form, 6, t("dlg.settings.vad_silence"), self._silence_spin)
        self._engine_var.trace_add("write", lambda *_: self._update_silence_state())
        self._update_silence_state()

        prompt_wrap = tk.Frame(form, bg=C_SURFACE)
        prompt_wrap.columnconfigure(0, weight=1)
        border = tk.Frame(
            prompt_wrap,
            bg="#eef2f7",
            highlightbackground=C_BORDER,
            highlightthickness=1,
            bd=0,
        )
        border.grid(row=0, column=0, sticky="ew")
        self._prompt_text = tk.Text(
            border,
            wrap=tk.WORD,
            height=4,
            font=F_SMALL,
            bg="#fbfdff",
            fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=7,
        )
        self._prompt_text.pack(padx=1, pady=1)
        self._prompt_text.insert(tk.END, self._settings.prompts[self._settings.language])
        self._add_row(form, 7, t("dlg.settings.prompt"), prompt_wrap, sticky="ew")

        self._lang_var.trace_add("write", self._on_lang_change)

        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=2, column=0, sticky="e", pady=(10, 0))
        make_btn(btn_row, t("dlg.settings.reset"), self._reset).pack(side=tk.LEFT, padx=(0, 8))
        make_btn(btn_row, t("dlg.settings.save"), self._save, primary=True).pack(side=tk.LEFT)

    def _apply_initial_size(self) -> None:
        self._parent.update_idletasks()
        parent_w = max(self._parent.winfo_width(), 520)
        parent_h = max(self._parent.winfo_height(), 360)
        width = max(520, min(760, int(parent_w * 0.7)))
        height = max(360, min(620, int(parent_h * 0.78)))
        self._win.geometry(f"{width}x{height}")

    def _make_combo(self, parent: tk.Widget, variable: tk.StringVar, values: list[str], *, width: int) -> ttk.Combobox:
        return ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
            font=F_LABEL,
            style="Modern.TCombobox",
        )

    def _add_row(self, parent: tk.Widget, row: int, label: str, widget: tk.Widget, *, sticky: str = "w") -> None:
        tk.Label(
            parent,
            text=label,
            bg=C_SURFACE,
            fg=C_MUTED,
            font=self._META_FONT,
        ).grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=(0, 8))
        widget.grid(row=row, column=1, sticky=sticky, pady=(0, 8))

    def _add_toggle_pair(
        self,
        parent: tk.Widget,
        toggles: list[tuple[str, tk.BooleanVar]],
    ) -> None:
        for index, (label, variable) in enumerate(toggles):
            self._make_toggle_chip(parent, label, variable).pack(
                side=tk.TOP,
                anchor="w",
                pady=(0, 8) if index < len(toggles) - 1 else (0, 0),
            )

    def _make_toggle_chip(self, parent: tk.Widget, label: str, variable: tk.BooleanVar) -> tk.Frame:
        chip = tk.Frame(
            parent,
            bg=C_SURFACE,
            highlightbackground=C_BORDER,
            highlightthickness=1,
            bd=0,
            cursor="hand2",
            padx=10,
            pady=7,
        )

        dot = tk.Canvas(
            chip,
            width=18,
            height=18,
            bg=C_SURFACE,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        dot.pack(side=tk.LEFT)

        text = tk.Label(
            chip,
            text=label,
            bg=C_SURFACE,
            fg=C_TEXT,
            font=F_LABEL,
            cursor="hand2",
        )
        text.pack(side=tk.LEFT, padx=(8, 0))

        def redraw(*_) -> None:
            enabled = bool(variable.get())
            fill = "#eff6ff" if enabled else "#f8fafc"
            chip.config(
                bg=fill,
                highlightbackground=C_ACCENT if enabled else C_BORDER,
            )
            dot.config(bg=fill)
            text.config(bg=fill)
            dot.delete("all")
            if enabled:
                dot.create_oval(1, 1, 17, 17, fill=C_ACCENT, outline=C_ACCENT)
                dot.create_text(9, 9, text="✓", fill="#ffffff", font=("TkDefaultFont", 9, "bold"))
            else:
                dot.create_oval(1, 1, 17, 17, fill=C_SURFACE, outline=C_BORDER, width=1)

        def toggle(_event=None) -> None:
            variable.set(not variable.get())
            redraw()

        for widget in (chip, dot, text):
            widget.bind("<Button-1>", toggle)

        variable.trace_add("write", redraw)
        redraw()
        return chip

    def _update_silence_state(self) -> None:
        engine_key = next((k for k, v in ENGINE_LABELS.items() if v == self._engine_var.get()), "whisperlive")
        self._silence_spin.config(state="normal" if engine_key == "faster_whisper" else "disabled")

    @staticmethod
    def _speed_key(label: str) -> str:
        return next((s for s in ("fast", "normal") if t(f"speed.{s}") == label), "fast")

    def _on_lang_change(self, *_) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, self._settings.prompts[lang])

    def _reset(self) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, DEFAULT_PROMPTS[lang])

    def _save(self) -> None:
        lang = self._lang_var.get()
        speed = self._speed_key(self._speed_var.get())
        ui_lang = next((k for k, v in UI_LANGUAGES.items() if v == self._ui_lang_var.get()), "en")
        compute_device = self._device_var.get()
        engine_type = next((k for k, v in ENGINE_LABELS.items() if v == self._engine_var.get()), "whisperlive")
        new_prompts = {**self._settings.prompts, lang: self._prompt_text.get("1.0", tk.END).strip()}
        try:
            silence_gap = round(max(0.1, min(10.0, float(self._silence_var.get()))), 1)
        except ValueError:
            silence_gap = 0.8
        self._on_save(replace(
            self._settings,
            language=lang,
            prompts=new_prompts,
            model_speed=speed,
            ui_language=ui_lang,
            compute_device=compute_device,
            asr_postprocess=self._asr_var.get(),
            cleanup_recordings=self._cleanup_var.get(),
            engine_type=engine_type,
            vad_silence_gap=silence_gap,
        ))
        self._win.destroy()
