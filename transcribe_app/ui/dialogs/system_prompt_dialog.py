import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from typing import Callable

from ..theme import C_BG, C_SURFACE, C_TEXT, C_BORDER, C_MUTED, F_SMALL, F_LABEL, make_btn
from transcribe_app.config import LANGUAGE_OPTS, DEFAULT_SYSTEM_PROMPTS
from transcribe_app.settings import Settings


class SystemPromptDialog:
    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save  = on_save

        self._win = tk.Toplevel(parent)
        self._win.title("System Prompt – Rechtschreibkorrektur")
        self._win.minsize(500, 340)
        self._win.configure(bg=C_BG)
        self._win.grab_set()
        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        top = tk.Frame(outer, bg=C_BG)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(top, text="Sprache:", bg=C_BG, fg=C_MUTED, font=F_SMALL).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self._lang_var = tk.StringVar(value=self._settings.language)
        lang_combo = ttk.Combobox(
            top,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly",
            width=12,
            font=F_LABEL,
        )
        lang_combo.pack(side=tk.LEFT)
        lang_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        border = tk.Frame(outer, bg=C_BORDER)
        border.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        border.rowconfigure(0, weight=1)
        border.columnconfigure(0, weight=1)

        self._prompt_text = tk.Text(
            border,
            wrap=tk.WORD, font=F_SMALL,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, bd=0, padx=10, pady=8,
        )
        self._prompt_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._prompt_text.insert(tk.END, self._settings.system_prompts[self._settings.language])

        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=2, column=0, sticky="e")
        make_btn(btn_row, "Zurücksetzen", self._reset).pack(side=tk.LEFT, padx=(0, 8))
        make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

    def _on_lang_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, self._settings.system_prompts[lang])

    def _reset(self) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, DEFAULT_SYSTEM_PROMPTS[lang])

    def _save(self) -> None:
        lang = self._lang_var.get()
        new_system_prompts = {**self._settings.system_prompts,
                               lang: self._prompt_text.get("1.0", tk.END).strip()}
        self._on_save(replace(self._settings, system_prompts=new_system_prompts))
        self._win.destroy()
