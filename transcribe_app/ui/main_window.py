"""TranscriptionApp — Tkinter view + event wiring.

Responsibilities
----------------
* Build and own all Tkinter widgets.
* Hold one Settings instance; delegate persistence to settings.save().
* Delegate engine/audio concerns to EngineManager.
* Bridge the asyncio → Tk thread boundary via a queue.Queue polled every 50 ms.

Not responsible for
-------------------
* Model loading, audio streaming, asyncio event loop  →  EngineManager
* Text cleaning / voice commands                      →  text_processing
"""

import queue
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from transcribe_app.config import LANGUAGE_OPTS
from ..engine import EngineManager, loading_status
from .. import settings as settings_io
from transcribe_app.settings import Settings
from ..text_processing import apply_commands, clean, strip_prompt_leak
from .theme import (
    C_ACCENT, C_ACCENT_H, C_BG, C_BORDER, C_BUFFER, C_DANGER, C_DANGER_H,
    C_HEADER, C_MUTED, C_STATUS_BG, C_SURFACE, C_TEXT,
    F_MONO, F_SMALL, apply_ttk_style, hoverable, make_btn,
)


class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live Transcription")
        self.root.minsize(560, 460)
        self.root.configure(bg=C_BG)

        self._settings: Settings    = settings_io.load()
        self._recording: bool       = False
        self._space_held: bool      = False   # push-to-talk state
        self._ui_queue: queue.Queue = queue.Queue()

        # Display state
        self._session_prefix:     str   = ""
        self._last_raw_committed: str   = ""
        self._absorbed_committed: str   = ""
        self._last_text_time:     float = 0.0
        self._last_display_sig:   str   = ""

        self._mgr = EngineManager(
            on_status=lambda msg: self._ui_queue.put(("status", msg)),
            on_ready=lambda: self._ui_queue.put(("enable_controls",)),
            on_update=lambda c, b: self._ui_queue.put(("update", c, b)),
            on_finalise=lambda c: self._ui_queue.put(("finalise", c)),
            on_open_mic=lambda: self.root.after(
                0, lambda: self._mgr.open_mic_stream(self._settings.input_device)
            ),
        )

        apply_ttk_style(root)
        self._build_ui()
        self._mgr.start(self._settings.language, self._settings.prompts[self._settings.language])
        self._poll_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Menu bar
        menubar = tk.Menu(self.root, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Save as file…", command=self._save_file)
        file_menu.add_command(label="Laden…",     command=self._load_file)

        edit_menu = tk.Menu(menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        edit_menu.add_command(label="Stimme & Style Prompt…", command=self._open_voice_style)

        # Row 0 — header
        header = tk.Frame(self.root, bg=C_HEADER)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=0, column=0, sticky="sew")

        tk.Label(
            header, text="Live Transcription",
            bg=C_HEADER, fg=C_TEXT,
            font=("TkDefaultFont", 13, "bold"),
            padx=16, pady=12,
        ).grid(row=0, column=0, sticky="w")

        lang_frame = tk.Frame(header, bg=C_HEADER)
        lang_frame.grid(row=0, column=2, padx=12, pady=8, sticky="e")
        tk.Label(lang_frame, text="Sprache", bg=C_HEADER, fg=C_MUTED, font=F_SMALL).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._lang_var = tk.StringVar(value=self._settings.language)
        self._lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly", width=10,
            font=("TkDefaultFont", 10),
        )
        self._lang_combo.pack(side=tk.LEFT)
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        # Row 1 — text area
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=1, column=0, sticky="new")
        border_frame = tk.Frame(self.root, bg=C_BORDER)
        border_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        border_frame.rowconfigure(0, weight=1)
        border_frame.columnconfigure(0, weight=1)

        self._text = scrolledtext.ScrolledText(
            border_frame,
            wrap=tk.WORD, font=F_MONO,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            selectbackground=C_ACCENT, selectforeground="#ffffff",
            relief=tk.FLAT, bd=0, padx=14, pady=12,
            undo=True,
        )
        self._text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._text.bind("<Control-a>", self._select_all)
        self._text.tag_configure("buffer",    foreground=C_BUFFER,
                                              font=("TkFixedFont", 11, "italic"))
        self._text.tag_configure("h1",        font=("TkDefaultFont", 17, "bold"),
                                              foreground=C_ACCENT, spacing1=10, spacing3=4)
        self._text.tag_configure("h2",        font=("TkDefaultFont", 13, "bold"),
                                              foreground=C_TEXT,   spacing1=8,  spacing3=2)
        self._text.tag_configure("h3",        font=("TkDefaultFont", 11, "bold"),
                                              foreground=C_MUTED,  spacing1=4)
        self._text.tag_configure("para_space", spacing1=10)

        # Row 2 — button bar
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=2, column=0, sticky="ew")
        btn_bar = tk.Frame(self.root, bg=C_BG, pady=10)
        btn_bar.grid(row=2, column=0, sticky="ew")
        btn_bar.columnconfigure((0, 1, 2, 3), weight=1)

        self._record_btn = make_btn(btn_bar, "⏺  Record", self._toggle_recording, primary=True)
        self._record_btn.config(state=tk.DISABLED)
        self._record_btn.grid(row=0, column=0, padx=(12, 6))

        make_btn(btn_bar, "Clear",    self._clear_text   ).grid(row=0, column=1, padx=6)
        make_btn(btn_bar, "Copy",     self._copy_text    ).grid(row=0, column=2, padx=6)
        make_btn(btn_bar, "Test Mic", self._open_mic_test).grid(row=0, column=3, padx=(6, 12))

        # Row 3/4 — status bar
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=3, column=0, sticky="ew")
        self._status_var = tk.StringVar(
            value=loading_status(
                LANGUAGE_OPTS[self._settings.language]["model_size"],
                self._settings.language,
            )
        )
        tk.Label(
            self.root, textvariable=self._status_var,
            anchor="w", padx=12, pady=5,
            bg=C_STATUS_BG, fg=C_MUTED, font=F_SMALL,
        ).grid(row=4, column=0, sticky="ew")

        self.root.config(cursor="watch")

        # Push-to-talk: hold Space to record, release to stop
        self.root.bind("<KeyPress-space>",   self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)

    # ── Language selector ──────────────────────────────────────────────────────

    def _on_language_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        if lang == self._settings.language:
            return
        self._settings = replace(self._settings, language=lang)
        settings_io.save(self._settings)
        self._reload_engine(lang)

    def _reload_engine(self, lang: str) -> None:
        if self._recording:
            self._stop_recording()
        self._record_btn.config(state=tk.DISABLED)
        self._status_var.set(loading_status(LANGUAGE_OPTS[lang]["model_size"], lang))
        self.root.config(cursor="watch")
        self._mgr.reload(lang, self._settings.prompts[lang])

    # ── Recording ──────────────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_space_press(self, event: tk.Event) -> None:
        """Start recording while Space is held (push-to-talk)."""
        if self._space_held or self._recording:
            return  # ignore key-repeat or already recording via button
        if str(self._record_btn.cget("state")) == tk.DISABLED:
            return
        self._space_held = True
        self._start_recording()

    def _on_space_release(self, event: tk.Event) -> None:
        """Stop recording when Space is released."""
        if not self._space_held:
            return
        self._space_held = False
        if self._recording:
            self._stop_recording()

    def _start_recording(self) -> None:
        self._recording = True
        self._record_btn.config(
            text="⏹  Stop",
            bg=C_DANGER, activebackground=C_DANGER_H,
        )
        hoverable(self._record_btn, C_DANGER, C_DANGER_H)
        self._status_var.set("Recording…")
        self._text.config(state=tk.DISABLED)
        self._text.edit_reset()

        existing = self._text.get("1.0", tk.END).rstrip("\n")
        self._session_prefix     = (existing + "\n") if existing else ""
        self._last_raw_committed  = ""
        self._absorbed_committed  = ""
        self._last_text_time      = time.monotonic()
        self._last_display_sig    = ""

        self._mgr.start_session()

    def _stop_recording(self) -> None:
        self._recording = False
        self._record_btn.config(
            text="⏺  Record",
            bg=C_ACCENT, activebackground=C_ACCENT_H,
        )
        hoverable(self._record_btn, C_ACCENT, C_ACCENT_H)
        self._status_var.set("Processing remaining audio…")
        self._text.config(state=tk.NORMAL)
        self._text.edit_reset()
        self._mgr.stop_session()

    # ── UI queue polling ───────────────────────────────────────────────────────

    def _poll_ui(self) -> None:
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                match msg[0]:
                    case "status":
                        self._status_var.set(msg[1])
                    case "enable_controls":
                        self._record_btn.config(state=tk.NORMAL)
                        self.root.config(cursor="")
                    case "update":
                        _, committed, buffer = msg
                        self._set_text(committed, buffer)
                    case "finalise":
                        _, committed = msg
                        self._set_text(committed, "")
        except queue.Empty:
            pass

        # Auto-restructure after 5 s of silence while recording
        if (
            self._recording
            and self._last_text_time > 0.0
            and (time.monotonic() - self._last_text_time) > 5.0
        ):
            self._restructure()

        self.root.after(50, self._poll_ui)

    # ── Text area helpers ──────────────────────────────────────────────────────

    def _begin_write(self) -> None:
        """Temporarily re-enable the text widget for a programmatic update."""
        self._text.config(state=tk.NORMAL)

    def _end_write(self) -> None:
        """Re-disable the text widget if recording is active."""
        if self._recording:
            self._text.config(state=tk.DISABLED)

    # ── Text display ───────────────────────────────────────────────────────────

    def _set_text(self, committed: str, buffer: str) -> None:
        self._last_raw_committed = committed

        display_committed = committed
        if self._absorbed_committed:
            if committed.startswith(self._absorbed_committed):
                display_committed = committed[len(self._absorbed_committed):]
            else:
                self._absorbed_committed = ""

        prompt = self._settings.prompts[self._settings.language]
        display_committed = apply_commands(clean(strip_prompt_leak(display_committed, prompt)))
        display_buffer    = clean(strip_prompt_leak(buffer, prompt))

        sig = display_committed + "\x00" + display_buffer
        if sig != self._last_display_sig:
            self._last_display_sig = sig
            self._last_text_time   = time.monotonic()

        self._begin_write()
        self._text.delete("1.0", tk.END)
        if self._session_prefix:
            self._text.insert(tk.END, self._session_prefix)
        if display_committed:
            self._text.insert(tk.END, display_committed)
        if display_buffer:
            self._text.insert(tk.END, (" " if display_committed else "") + display_buffer, "buffer")
        self._render_markdown()
        self._end_write()
        self._text.see(tk.END)

    def _restructure(self) -> None:
        """Apply clean + apply_commands to the whole text area after a silence pause."""
        current = self._text.get("1.0", tk.END).rstrip()
        if not current:
            self._last_text_time = 0.0
            return
        restructured = apply_commands(clean(current))
        self._session_prefix     = restructured + "\n"
        self._absorbed_committed = self._last_raw_committed
        self._last_text_time     = 0.0
        self._last_display_sig   = ""

        self._begin_write()
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, self._session_prefix)
        self._render_markdown()
        self._end_write()
        self._text.see(tk.END)

    def _render_markdown(self) -> None:
        for tag in ("h1", "h2", "h3", "para_space"):
            self._text.tag_remove(tag, "1.0", tk.END)
        for i, line in enumerate(self._text.get("1.0", tk.END).split("\n"), start=1):
            if line.startswith("### "):
                self._text.tag_add("h3", f"{i}.0", f"{i}.end")
            elif line.startswith("## "):
                self._text.tag_add("h2", f"{i}.0", f"{i}.end")
            elif line.startswith("# "):
                self._text.tag_add("h1", f"{i}.0", f"{i}.end")
            elif line.strip() == "":
                self._text.tag_add("para_space", f"{i}.0", f"{i}.end")

    # ── Text area actions ──────────────────────────────────────────────────────

    def _select_all(self, _event=None):
        self._text.tag_add(tk.SEL, "1.0", tk.END)
        self._text.mark_set(tk.INSERT, "1.0")
        self._text.see(tk.INSERT)
        return "break"

    def _clear_text(self) -> None:
        self._begin_write()
        self._text.delete("1.0", tk.END)
        self._end_write()

    def _copy_text(self) -> None:
        content = self._text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._status_var.set("Copied to clipboard ✓")
        self.root.after(2000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._settings.language} model loaded"
        ))

    # ── File menu ──────────────────────────────────────────────────────────────

    def _save_file(self) -> None:
        text = self._text.get("1.0", tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Textdatei", "*.txt"), ("Markdown", "*.md"), ("Alle Dateien", "*")],
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")
        self._status_var.set(f"Gespeichert: {Path(path).name}")
        self.root.after(3000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._settings.language} model loaded"
        ))

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Textdatei", "*.txt"), ("Markdown", "*.md"), ("Alle Dateien", "*")],
        )
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8")
        self._begin_write()
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, text)
        self._end_write()
        self._session_prefix     = text
        self._last_raw_committed  = ""
        self._absorbed_committed  = ""
        self._last_display_sig    = ""
        self._last_text_time      = 0.0
        self._render_markdown()
        self._status_var.set(f"Geladen: {Path(path).name}")
        self.root.after(3000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._settings.language} model loaded"
        ))

    # ── Edit menu dialogs ──────────────────────────────────────────────────────

    def _open_voice_style(self) -> None:
        from .dialogs.voice_style_dialog import VoiceStyleDialog

        def on_save(new: Settings) -> None:
            old_lang   = self._settings.language
            old_prompt = self._settings.prompts[old_lang]
            self._settings = new
            settings_io.save(self._settings)
            self._lang_var.set(new.language)
            if new.language != old_lang or new.prompts[new.language] != old_prompt:
                self._reload_engine(new.language)

        VoiceStyleDialog(self.root, self._settings, on_save)

    def _open_mic_test(self) -> None:
        from .mic_test import MicTestWindow

        def _on_device_save(new: Settings) -> None:
            self._settings = new
            settings_io.save(self._settings)

        MicTestWindow(self.root, self._settings, _on_device_save)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._mgr.shutdown()
        self.root.destroy()
