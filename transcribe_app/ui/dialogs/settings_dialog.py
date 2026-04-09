import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..theme import C_BG, C_MUTED, F_SMALL, F_LABEL, make_btn
from transcribe_app.config import CORRECTION_BACKENDS
from transcribe_app.settings import Settings


class SettingsDialog:
    _DESCRIPTIONS = {
        "Ollama": (
            "Nutzt das lokal laufende Ollama-LLM. Kontextsensitiv, "
            "korrigiert Grammatik und Interpunktion. "
            "Erfordert einen laufenden Ollama-Server."
        ),
        "LanguageTool": (
            "Regelbasierte Grammatik- und Rechtschreibprüfung. "
            "Sehr gut für Deutsch. Erfordert Java und "
            "pip install language_tool_python."
        ),
        "Spylls": (
            "Wörterbuchbasierte Rechtschreibkorrektur (Hunspell). "
            "Kein Java, kein Server. Nur Wortebene — keine Grammatik. "
            "Erfordert pip install spylls sowie installierte "
            "Hunspell-Wörterbücher (z. B. hunspell-de / hunspell-en-us)."
        ),
    }

    def __init__(
        self,
        parent: tk.Widget,
        settings: Settings,
        on_save: Callable[[Settings], None],
    ) -> None:
        self._settings = settings
        self._on_save  = on_save

        self._win = tk.Toplevel(parent)
        self._win.title("Einstellungen")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.grab_set()
        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        tk.Label(
            outer, text="Rechtschreibkorrektur-Backend:",
            bg=C_BG, fg=C_MUTED, font=F_SMALL,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 6))

        self._backend_var = tk.StringVar(value=self._settings.correction_backend)
        ttk.Combobox(
            outer,
            textvariable=self._backend_var,
            values=CORRECTION_BACKENDS,
            state="readonly",
            width=16,
            font=F_LABEL,
        ).grid(row=0, column=1, sticky="w", pady=(0, 6))

        self._desc_var = tk.StringVar()
        tk.Label(
            outer, textvariable=self._desc_var,
            bg=C_BG, fg=C_MUTED, font=F_SMALL,
            wraplength=320, justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self._backend_var.trace_add("write", self._update_desc)
        self._update_desc()

        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="e")
        make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

    def _update_desc(self, *_) -> None:
        self._desc_var.set(self._DESCRIPTIONS.get(self._backend_var.get(), ""))

    def _save(self) -> None:
        from dataclasses import replace
        self._on_save(replace(self._settings, correction_backend=self._backend_var.get()))
        self._win.destroy()
