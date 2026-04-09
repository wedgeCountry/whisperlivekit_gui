import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from typing import Callable

from ..theme import C_BG, C_SURFACE, C_TEXT, C_BORDER, C_MUTED, F_SMALL, F_LABEL, make_btn, center_on_parent
from transcribe_app.config import LANGUAGE_OPTS, DEFAULT_PROMPTS
from transcribe_app.settings import Settings


class VoiceStyleDialog:
    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save  = on_save

        self._win = tk.Toplevel(parent)
        self._win.title("Stimme & Style Prompt")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.grab_set()
        self._build_ui()
        center_on_parent(self._win, parent)

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        tk.Label(outer, text="Stimme:", bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
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

        tk.Label(outer, text="Style Prompt:", bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=1, column=0, sticky="nw", padx=(0, 10), pady=(0, 12)
        )
        border = tk.Frame(outer, bg=C_BORDER)
        border.grid(row=1, column=1, sticky="ew", pady=(0, 12))

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
        btn_row.grid(row=2, column=0, columnspan=2, sticky="e")
        make_btn(btn_row, "Zurücksetzen", self._reset).pack(side=tk.LEFT, padx=(0, 8))
        make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

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
        new_prompts = {**self._settings.prompts, lang: self._prompt_text.get("1.0", tk.END).strip()}
        self._on_save(replace(self._settings, language=lang, prompts=new_prompts))
        self._win.destroy()
