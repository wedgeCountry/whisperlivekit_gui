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

from transcribe_app.config import LANGUAGE_OPTS, get_model_size, SPACE_HOLD_TIME_MS
from ..engine import EngineManager, loading_status
from ..i18n import UI_LANGUAGES, get_language, set_language, t
from .. import settings as settings_io
from transcribe_app.settings import Settings
from ..text_processing import apply_commands_full, clean, strip_prompt_leak
from .theme import (
    C_ACCENT, C_ACCENT_H, C_BG, C_BORDER, C_BUFFER, C_DANGER, C_DANGER_H,
    C_HEADER, C_MUTED, C_STATUS_BG, C_SURFACE, C_TEXT,
    F_MONO, F_SMALL, apply_ttk_style, hoverable, make_btn,
)


class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(t("window.title"))
        self.root.minsize(640, 460)
        self.root.configure(bg=C_BG)

        self._settings: Settings    = settings_io.load()
        set_language(self._settings.ui_language)
        self._recording: bool       = False
        self._space_held: bool      = False   # push-to-talk state
        self._space_after_id: str | None = None  # pending 3-s timer
        self._popup: tk.Toplevel | None = None  # at most one popup open at a time
        self._ui_queue: queue.Queue = queue.Queue()

        # Display state
        self._session_prefix:     str   = ""
        self._session_suffix:     str   = ""   # text after cursor, preserved during recording
        self._recording_prefix:   str   = ""   # snapshot of _session_prefix at recording start
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
        self._mgr.start(
            self._settings.language,
            self._settings.prompts[self._settings.language],
            self._settings.model_speed,
        )
        self._poll_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Menu bar
        self._menubar = tk.Menu(self.root, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self.root.config(menu=self._menubar)

        self._file_menu = tk.Menu(self._menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self._menubar.add_cascade(label=t("menu.file"), menu=self._file_menu)
        self._file_menu.add_command(label=t("menu.file.save"), command=self._save_file)
        self._file_menu.add_command(label=t("menu.file.load"), command=self._load_file)

        self._edit_menu = tk.Menu(self._menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self._menubar.add_cascade(label=t("menu.edit"), menu=self._edit_menu)
        self._edit_menu.add_command(label=t("menu.edit.voice_style"), command=self._open_voice_style)

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
        self._lang_label = tk.Label(
            lang_frame, text=t("header.language"), bg=C_HEADER, fg=C_MUTED, font=F_SMALL
        )
        self._lang_label.pack(side=tk.LEFT, padx=(0, 6))
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

        speed_frame = tk.Frame(header, bg=C_HEADER)
        speed_frame.grid(row=0, column=3, padx=(0, 4), pady=8, sticky="e")
        self._model_label = tk.Label(
            speed_frame, text=t("header.model"), bg=C_HEADER, fg=C_MUTED, font=F_SMALL
        )
        self._model_label.pack(side=tk.LEFT, padx=(0, 6))
        self._speed_var = tk.StringVar(value=self._settings.model_speed)
        self._speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self._speed_var,
            values=["fast", "normal"],
            state="readonly", width=7,
            font=("TkDefaultFont", 10),
        )
        self._speed_combo.pack(side=tk.LEFT)
        self._speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        ui_frame = tk.Frame(header, bg=C_HEADER)
        ui_frame.grid(row=0, column=4, padx=(0, 12), pady=8, sticky="e")
        self._ui_label = tk.Label(
            ui_frame, text=t("header.interface"), bg=C_HEADER, fg=C_MUTED, font=F_SMALL
        )
        self._ui_label.pack(side=tk.LEFT, padx=(0, 6))
        self._ui_lang_var = tk.StringVar(value=UI_LANGUAGES[self._settings.ui_language])
        self._ui_lang_combo = ttk.Combobox(
            ui_frame,
            textvariable=self._ui_lang_var,
            values=list(UI_LANGUAGES.values()),
            state="readonly", width=8,
            font=("TkDefaultFont", 10),
        )
        self._ui_lang_combo.pack(side=tk.LEFT)
        self._ui_lang_combo.bind("<<ComboboxSelected>>", self._on_ui_lang_change)

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

        self._record_btn = make_btn(btn_bar, t("btn.record"), self._toggle_recording, primary=True)
        self._record_btn.config(state=tk.DISABLED)
        self._record_btn.grid(row=0, column=0, padx=(12, 6))

        self._clear_btn = make_btn(btn_bar, t("btn.clear"),    self._clear_text)
        self._clear_btn.grid(row=0, column=1, padx=6)
        self._copy_btn = make_btn(btn_bar, t("btn.copy"),     self._copy_text)
        self._copy_btn.grid(row=0, column=2, padx=6)
        self._mic_test_btn = make_btn(btn_bar, t("btn.mic_test"), self._open_mic_test)
        self._mic_test_btn.grid(row=0, column=3, padx=(6, 12))

        # Row 3/4 — status bar
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=3, column=0, sticky="ew")
        self._status_var = tk.StringVar(
            value=loading_status(
                get_model_size(self._settings.language, self._settings.model_speed),
                self._settings.language,
            )
        )
        tk.Label(
            self.root, textvariable=self._status_var,
            anchor="w", padx=12, pady=5,
            bg=C_STATUS_BG, fg=C_MUTED, font=F_SMALL,
        ).grid(row=4, column=0, sticky="ew")

        self.root.config(cursor="watch")

        # Push-to-talk: hold Space ≥ 3 s to record; shorter press inserts a space
        self._text.bind("<KeyPress-space>",  self._on_space_press)   # intercept before Text class binding
        self.root.bind("<KeyPress-space>",   self._on_space_press)   # catch when text not focused
        self.root.bind("<KeyRelease-space>", self._on_space_release)

    # ── Language selector ──────────────────────────────────────────────────────

    def _on_language_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        if lang == self._settings.language:
            return
        self._settings = replace(self._settings, language=lang)
        settings_io.save(self._settings)
        self._reload_engine(lang)

    def _on_speed_change(self, _event=None) -> None:
        speed = self._speed_var.get()
        if speed == self._settings.model_speed:
            return
        self._settings = replace(self._settings, model_speed=speed)
        settings_io.save(self._settings)
        self._reload_engine(self._settings.language)

    def _on_ui_lang_change(self, _event=None) -> None:
        display = self._ui_lang_var.get()
        code = next((k for k, v in UI_LANGUAGES.items() if v == display), "en")
        if code == self._settings.ui_language:
            return
        self._settings = replace(self._settings, ui_language=code)
        settings_io.save(self._settings)
        set_language(code)
        self._apply_ui_lang()

    def _apply_ui_lang(self) -> None:
        """Re-render all static UI text in the active interface language."""
        self.root.title(t("window.title"))
        # Menu bar cascades
        self._menubar.entryconfigure(0, label=t("menu.file"))
        self._menubar.entryconfigure(1, label=t("menu.edit"))
        # File menu entries
        self._file_menu.entryconfigure(0, label=t("menu.file.save"))
        self._file_menu.entryconfigure(1, label=t("menu.file.load"))
        # Edit menu entries
        self._edit_menu.entryconfigure(0, label=t("menu.edit.voice_style"))
        # Header labels
        self._lang_label.config(text=t("header.language"))
        self._model_label.config(text=t("header.model"))
        self._ui_label.config(text=t("header.interface"))
        # Buttons (only record btn text depends on recording state)
        if not self._recording:
            self._record_btn.config(text=t("btn.record"))
        self._clear_btn.config(text=t("btn.clear"))
        self._copy_btn.config(text=t("btn.copy"))
        self._mic_test_btn.config(text=t("btn.mic_test"))

    def _reload_engine(self, lang: str) -> None:
        if self._recording:
            self._stop_recording()
        self._record_btn.config(state=tk.DISABLED)
        self._status_var.set(loading_status(get_model_size(lang, self._settings.model_speed), lang))
        self.root.config(cursor="watch")
        self._mgr.reload(lang, self._settings.prompts[lang], self._settings.model_speed)

    # ── Recording ──────────────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_space_press(self, event: tk.Event) -> str | None:
        """Arm the 3-second push-to-talk timer on Space press."""
        if self._space_held or self._recording:
            return "break"  # suppress key-repeat or extra presses during recording
        if str(self._record_btn.cget("state")) == tk.DISABLED:
            return None  # model not ready — let space through normally
        self._space_held = True
        self._space_after_id = self.root.after(SPACE_HOLD_TIME_MS, self._on_space_record)
        return "break"  # prevent text widget from inserting a space immediately

    def _on_space_record(self) -> None:
        """Called after 3 continuous seconds — start recording."""
        self._space_after_id = None
        if self._space_held:
            self._start_recording()

    def _on_space_release(self, event: tk.Event) -> None:
        """On early release (< 3 s): insert a space.  After 3 s: stop recording."""
        if not self._space_held:
            return
        self._space_held = False
        if self._space_after_id is not None:
            # Released before 3 s — cancel timer and insert a plain space
            self.root.after_cancel(self._space_after_id)
            self._space_after_id = None
            self._begin_write()
            self._text.insert(tk.INSERT, " ")
            self._end_write()
        elif self._recording:
            self._stop_recording()

    def _start_recording(self) -> None:
        self._recording = True
        self._record_btn.config(
            text=t("btn.stop"),
            bg=C_DANGER, activebackground=C_DANGER_H,
        )
        hoverable(self._record_btn, C_DANGER, C_DANGER_H)
        self._status_var.set(t("status.recording"))

        # Capture cursor position before disabling the widget
        cursor = self._text.index(tk.INSERT)
        self._session_prefix    = self._text.get("1.0", cursor)
        self._recording_prefix  = self._session_prefix   # snapshot for post-processing
        self._session_suffix    = self._text.get(cursor, "end-1c")
        self._last_raw_committed = ""
        self._absorbed_committed = ""
        self._last_text_time     = time.monotonic()
        self._last_display_sig   = ""

        self._mgr.mic_gain = self._settings.mic_gain
        self._text.config(state=tk.DISABLED)
        self._text.edit_reset()
        self._mgr.start_session()

    def _stop_recording(self) -> None:
        self._recording = False
        self._record_btn.config(
            text=t("btn.record"),
            bg=C_ACCENT, activebackground=C_ACCENT_H,
            state=tk.DISABLED,
        )
        hoverable(self._record_btn, C_ACCENT, C_ACCENT_H)
        self._status_var.set(t("status.processing"))
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
                        self._postprocess()
                        self._record_btn.config(state=tk.NORMAL)
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
        display_committed = clean(strip_prompt_leak(display_committed, prompt))
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
        if self._session_suffix:
            self._text.insert(tk.END, self._session_suffix)
        self._render_markdown()
        self._end_write()
        self._text.see(tk.END)

    def _restructure(self) -> None:
        """Bake accumulated text into _session_prefix to keep display_committed short.

        Voice commands are no longer applied here; they run in bulk via
        _postprocess() the moment recording ends.
        """
        full = self._text.get("1.0", tk.END)
        if self._session_suffix:
            idx = full.rfind(self._session_suffix)
            current = (full[:idx] if idx >= 0 else full).rstrip()
        else:
            current = full.rstrip()
        if not current:
            self._last_text_time = 0.0
            return
        self._session_prefix     = current + "\n"
        self._absorbed_committed = self._last_raw_committed
        self._last_text_time     = 0.0
        self._last_display_sig   = ""
        # No redraw: the text content is unchanged; the new structure takes
        # effect automatically on the next _set_text() call.

    def _postprocess(self) -> None:
        """Apply voice commands to the dictated text after recording ends.

        Only the text added during the current recording session is processed —
        any pre-existing content in the editor is left untouched.
        """
        full = self._text.get("1.0", "end-1c")

        # Split off the preserved suffix (text that was after the cursor).
        if self._session_suffix and self._session_suffix in full:
            idx  = full.rfind(self._session_suffix)
            body = full[:idx]
            suffix = full[idx:]
        else:
            body   = full
            suffix = ""

        # Identify just the newly dictated part (everything after the pre-recording prefix).
        prefix = self._recording_prefix
        if prefix and body.startswith(prefix):
            new_text = body[len(prefix):]
        else:
            # Safety fallback: no recognisable boundary — process the whole body.
            prefix   = ""
            new_text = body

        processed = apply_commands_full(clean(new_text))
        if processed is None:
            return  # nothing to substitute

        self._begin_write()
        self._text.delete("1.0", tk.END)
        if prefix:
            self._text.insert(tk.END, prefix)
        self._text.insert(tk.END, processed)
        if suffix:
            self._text.insert(tk.END, suffix)
        self._render_markdown()
        self._end_write()
        self._text.see(tk.END)

        # Keep session_prefix consistent so the next recording starts correctly.
        self._session_prefix = prefix + processed

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
        self._session_prefix = ""
        self._session_suffix = ""

    def _copy_text(self) -> None:
        content = self._text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._status_var.set(t("status.copied"))
        self.root.after(2000, lambda: self._status_var.set(
            t("status.recording") if self._recording
            else t("status.ready", lang=self._settings.language)
        ))

    # ── File menu ──────────────────────────────────────────────────────────────

    def _save_file(self) -> None:
        text = self._text.get("1.0", tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[(t("file.type.text"), "*.txt"), ("Markdown", "*.md"), (t("file.type.all"), "*")],
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")
        self._status_var.set(t("status.saved", name=Path(path).name))
        self.root.after(3000, lambda: self._status_var.set(
            t("status.recording") if self._recording
            else t("status.ready", lang=self._settings.language)
        ))

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[(t("file.type.text"), "*.txt"), ("Markdown", "*.md"), (t("file.type.all"), "*")],
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
        self._status_var.set(t("status.loaded", name=Path(path).name))
        self.root.after(3000, lambda: self._status_var.set(
            t("status.recording") if self._recording
            else t("status.ready", lang=self._settings.language)
        ))

    # ── Popup guard ────────────────────────────────────────────────────────────

    def _popup_open(self) -> bool:
        """Return True and raise the existing popup if one is already open."""
        if self._popup is not None and self._popup.winfo_exists():
            self._popup.lift()
            self._popup.focus_set()
            return True
        self._popup = None
        return False

    def _register_popup(self, win: tk.Toplevel) -> None:
        self._popup = win
        win.bind("<Destroy>", lambda _: setattr(self, "_popup", None), add=True)

    # ── Edit menu dialogs ──────────────────────────────────────────────────────

    def _open_voice_style(self) -> None:
        if self._popup_open():
            return
        from .dialogs.voice_style_dialog import VoiceStyleDialog

        def on_save(new: Settings) -> None:
            old_lang   = self._settings.language
            old_prompt = self._settings.prompts[old_lang]
            old_speed  = self._settings.model_speed
            self._settings = new
            settings_io.save(self._settings)
            self._lang_var.set(new.language)
            self._speed_var.set(new.model_speed)
            if (
                new.language != old_lang
                or new.prompts[new.language] != old_prompt
                or new.model_speed != old_speed
            ):
                self._reload_engine(new.language)

        dialog = VoiceStyleDialog(self.root, self._settings, on_save)
        self._register_popup(dialog._win)

    def _open_mic_test(self) -> None:
        if self._popup_open():
            return
        from .mic_test import MicTestWindow

        def _on_device_save(new: Settings) -> None:
            self._settings = new
            settings_io.save(self._settings)
            self._mgr.mic_gain = new.mic_gain

        win = MicTestWindow(self.root, self._settings, _on_device_save)
        self._register_popup(win._win)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._mgr.shutdown()
        self.root.destroy()
