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

import contextlib
import difflib
import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from ..session_file_manager import SessionFileManager
from ..recording_cleanup import start_async_recordings_cleanup

_log = logging.getLogger(__name__)

from transcribe_app.config import (
    DIFF_DIR, LANGUAGE_OPTS, get_model_size, SAMPLE_RATE, SESSIONS_DIR,
    SPACE_HOLD_TIME_MS, GPU, DTYPE,
)
from ..engine_protocol import EngineManagerProtocol, create_engine_manager
from ..i18n import set_language, t
from ..model_status import loading_status
from .. import settings as settings_io
from transcribe_app.settings import Settings
from ..text_processing import apply_commands_full, clean, strip_prompt_leak
from .theme import (
    C_ACCENT, C_ACCENT_H, C_BG, C_BORDER, C_BUFFER, C_DANGER, C_DANGER_H,
    C_HEADER, C_INPUT, C_MUTED, C_STATUS_BG, C_SURFACE, C_TEXT,
    F_MONO, F_SMALL, apply_ttk_style, hoverable, make_btn, make_card,
    set_button_enabled,
    style_text_widget,
)


class TranscriptionApp:
    _RECORDING_CLEANUP_MAX_AGE_S = 120.0

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
        self._retranscribing: bool          = False  # True while background re-transcription runs
        self._session_mgr:    "object | None" = None  # SessionFileManager for active/last session
        self._engine_ready:   bool          = False  # True once on_ready(True) arrives; False during reload
        self._session_draining: bool        = False  # True between stop_session() and _postprocess() completing
        self._pending_engine_action: "Callable[[], None] | None" = None  # deferred reload/recreate
        self._header_logo: "tk.PhotoImage | None" = None

        self._mgr: EngineManagerProtocol = self._make_manager()
        if hasattr(self._mgr, "vad_silence_gap"):
            self._mgr.vad_silence_gap = self._settings.vad_silence_gap
        if hasattr(self._mgr, "input_device"):
            self._mgr.input_device = self._settings.input_device

        apply_ttk_style(root)
        self._build_ui()
        self._mgr.start(
            self._settings.language,
            self._settings.prompts[self._settings.language],
            self._settings.model_speed,
            self._settings.compute_device,
        )
        self._poll_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Engine manager factory ─────────────────────────────────────────────────

    def _make_manager(self) -> EngineManagerProtocol:
        """Create the engine manager for the current settings.engine_type."""
        return create_engine_manager(
            self._settings.engine_type,
            on_status=lambda msg: self._ui_queue.put(("status", msg)),
            on_ready=lambda ok: self._ui_queue.put(("enable_controls", ok)),
            on_update=lambda c, b: self._ui_queue.put(("update", c, b)),
            on_finalise=lambda c: self._ui_queue.put(("finalise", c)),
            # on_open_mic references self._mgr by name so that after _recreate_manager
            # replaces self._mgr, the new manager's open_mic_stream is called.
            on_open_mic=lambda: self.root.after(0, self._open_mic_stream_from_ui),
        )

    def _open_mic_stream_from_ui(self) -> None:
        """Open the live mic stream and recover cleanly if Windows audio setup fails."""
        try:
            self._mgr.open_mic_stream(self._settings.input_device)
        except Exception as exc:  # noqa: BLE001
            _log.error("Opening live microphone stream failed", exc_info=True)
            self._mgr.audio_sink = None
            if self._session_mgr is not None:
                self._session_mgr.cleanup()
                self._session_mgr = None
            try:
                self._mgr.stop_session()
            except Exception:
                _log.warning("Stopping failed recording session after mic-open error failed", exc_info=True)
            self._recording = False
            self._session_draining = False
            self._record_btn.config(
                text=t("btn.record"),
                bg=C_ACCENT, activebackground=C_ACCENT_H,
            )
            hoverable(self._record_btn, C_ACCENT, C_ACCENT_H)
            set_button_enabled(self._clear_btn, True)
            self._text.config(state=tk.NORMAL)
            self._status_var.set(t("status.error", exc=exc))
            self._set_record_btn_state()
            self.root.config(cursor="")

    def _recreate_manager(self) -> None:
        """Shut down the current manager and replace it with a fresh one."""
        self._mgr.shutdown()
        self._engine_ready = False
        set_button_enabled(self._record_btn, False)
        self._mgr = self._make_manager()
        self._mgr.mic_gain = self._settings.mic_gain
        if hasattr(self._mgr, "vad_silence_gap"):
            self._mgr.vad_silence_gap = self._settings.vad_silence_gap
        if hasattr(self._mgr, "input_device"):
            self._mgr.input_device = self._settings.input_device
        self._mgr.start(
            self._settings.language,
            self._settings.prompts[self._settings.language],
            self._settings.model_speed,
            self._settings.compute_device,
        )

    def _attach_header_logo(self, parent: tk.Widget) -> None:
        """Show a small version of the app icon in the header when available."""
        logo_image = getattr(self.root, "_app_icon_image", None)
        if logo_image is None:
            return

        width = max(1, int(logo_image.width()))
        height = max(1, int(logo_image.height()))
        scale = max(1, max((width + 27) // 28, (height + 27) // 28))
        self._header_logo = logo_image.subsample(scale, scale)
        tk.Label(
            parent,
            image=self._header_logo,
            bg=C_HEADER,
            bd=0,
        ).pack(side=tk.LEFT, padx=(0, 10))

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Menu bar
        # _menu_labels: (menu, index, i18n_key) — built once, iterated by _apply_ui_lang
        self._menu_labels: list[tuple[tk.Menu, int, str]] = []
        self._menubar = tk.Menu(self.root, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self.root.config(menu=self._menubar)

        self._file_menu = tk.Menu(self._menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self._menubar.add_cascade(label=t("menu.file"), menu=self._file_menu)
        self._menu_labels.append((self._menubar, self._menubar.index(tk.END), "menu.file"))
        self._file_menu.add_command(label=t("menu.file.save"), command=self._save_file)
        self._menu_labels.append((self._file_menu, self._file_menu.index(tk.END), "menu.file.save"))
        self._file_menu.add_command(label=t("menu.file.load"), command=self._load_file)
        self._menu_labels.append((self._file_menu, self._file_menu.index(tk.END), "menu.file.load"))

        self._edit_menu = tk.Menu(self._menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self._menubar.add_cascade(label=t("menu.edit"), menu=self._edit_menu)
        self._menu_labels.append((self._menubar, self._menubar.index(tk.END), "menu.edit"))
        self._edit_menu.add_command(label=t("menu.edit.settings"), command=self._open_settings)
        self._menu_labels.append((self._edit_menu, self._edit_menu.index(tk.END), "menu.edit.settings"))

        # Row 0 — header
        header = make_card(self.root, bg=C_HEADER, padx=16, pady=12)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        header.columnconfigure(1, weight=1)

        title_wrap = tk.Frame(header, bg=C_HEADER)
        title_wrap.grid(row=0, column=0, rowspan=2, sticky="w")
        self._attach_header_logo(title_wrap)

        title_text = tk.Frame(title_wrap, bg=C_HEADER)
        title_text.pack(side=tk.LEFT, anchor="w")
        self._title_label = tk.Label(
            title_text, text=t("window.title"),
            bg=C_HEADER, fg=C_TEXT,
            font=("TkDefaultFont", 13, "bold"),
        )
        self._title_label.pack(anchor="w")
        self._subtitle_label = tk.Label(
            title_text,
            text=t("status.ready", lang=self._settings.language),
            bg=C_HEADER,
            fg=C_MUTED,
            font=F_SMALL,
        )
        self._subtitle_label.pack(anchor="w", pady=(3, 0))

        lang_frame = tk.Frame(header, bg=C_HEADER)
        lang_frame.grid(row=0, column=2, rowspan=2, padx=12, sticky="e")
        self._lang_label = tk.Label(
            lang_frame, text=t("header.language"), bg=C_HEADER, fg=C_MUTED, font=F_SMALL
        )
        self._lang_label.pack(anchor="w", padx=(2, 0))
        self._lang_var = tk.StringVar(value=self._settings.language)
        self._lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly", width=10,
            font=("TkDefaultFont", 10),
            style="Modern.TCombobox",
        )
        self._lang_combo.pack(anchor="w", pady=(4, 0))
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        speed_frame = tk.Frame(header, bg=C_HEADER)
        speed_frame.grid(row=0, column=3, rowspan=2, padx=(0, 4), sticky="e")
        self._model_label = tk.Label(
            speed_frame, text=t("header.model"), bg=C_HEADER, fg=C_MUTED, font=F_SMALL
        )
        self._model_label.pack(anchor="w", padx=(2, 0))
        self._speed_var = tk.StringVar(value=t(f"speed.{self._settings.model_speed}"))
        self._speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self._speed_var,
            values=[t("speed.fast"), t("speed.normal")],
            state="readonly", width=8,
            font=("TkDefaultFont", 10),
            style="Modern.TCombobox",
        )
        self._speed_combo.pack(anchor="w", pady=(4, 0))
        self._speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Row 1 — text area
        border_frame = make_card(self.root, padx=1, pady=1)
        border_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        border_frame.rowconfigure(0, weight=1)
        border_frame.columnconfigure(0, weight=1)

        self._text = scrolledtext.ScrolledText(
            border_frame,
            wrap=tk.WORD, font=F_MONO,
            padx=14, pady=12,
            undo=True,
        )
        self._text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        style_text_widget(self._text)
        self._text.configure(bg=C_INPUT)
        self._text.vbar.configure(
            bg="#cbd5e1",
            activebackground="#94a3b8",
            troughcolor=C_SURFACE,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            width=10,
        )
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
        btn_bar = tk.Frame(self.root, bg=C_BG, pady=10)
        btn_bar.grid(row=2, column=0, sticky="ew")
        btn_bar.columnconfigure((0, 1, 2, 3), weight=1)

        self._record_btn = make_btn(btn_bar, t("btn.record"), self._toggle_recording, primary=True)
        set_button_enabled(self._record_btn, False)
        self._record_btn.grid(row=0, column=0, padx=(12, 6))

        self._clear_btn = make_btn(btn_bar, t("btn.clear"),    self._clear_text)
        self._clear_btn.grid(row=0, column=1, padx=6)
        self._copy_btn = make_btn(btn_bar, t("btn.copy"),     self._copy_text)
        self._copy_btn.grid(row=0, column=2, padx=6)
        self._mic_test_btn = make_btn(btn_bar, t("btn.mic_test"), self._open_mic_test)
        self._mic_test_btn.grid(row=0, column=3, padx=(6, 12))

        # Row 3/4 — status bar
        _use_gpu = self._settings.compute_device == "cuda"
        self._status_var = tk.StringVar(
            value=loading_status(
                get_model_size(self._settings.language, self._settings.model_speed, _use_gpu),
                self._settings.language,
                _use_gpu,
            )
        )
        self._status_card = make_card(self.root, bg=C_STATUS_BG, border=C_BORDER, padx=12, pady=8)
        self._status_card.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
        tk.Label(
            self._status_card, textvariable=self._status_var,
            anchor="w",
            bg=C_STATUS_BG, fg=C_MUTED, font=F_SMALL,
        ).pack(fill=tk.X)

        self.root.config(cursor="watch")

        # Push-to-talk: hold Space ≥ 3 s to record; shorter press inserts a space
        self._text.bind("<KeyPress-space>",  self._on_space_press)   # intercept before Text class binding
        self.root.bind("<KeyPress-space>",   self._on_space_press)   # catch when text not focused
        self.root.bind("<KeyRelease-space>", self._on_space_release)

    def _set_record_btn_state(self) -> None:
        """Enable the record button only when engine is ready AND no session is draining."""
        if self._engine_ready and not self._session_draining:
            set_button_enabled(self._record_btn, True)
            set_button_enabled(self._clear_btn, True)

    def _finalise_session(self) -> bool:
        """Called when a session finishes draining.

        Runs any pending engine action (reload/recreate from a settings save that
        arrived while a session was active).  Returns True if an action ran — the
        caller should skip the 'ready' status update in that case.
        """
        action = self._pending_engine_action
        self._pending_engine_action = None
        if action is not None:
            action()
            return True
        self._set_record_btn_state()
        return False

    # ── Language selector ──────────────────────────────────────────────────────

    def _on_language_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        if lang == self._settings.language:
            return
        self._settings = replace(self._settings, language=lang)
        settings_io.save(self._settings)
        self._reload_engine(lang)

    def _on_speed_change(self, _event=None) -> None:
        label = self._speed_var.get()
        speed = next((s for s in ("fast", "normal") if t(f"speed.{s}") == label), None)
        if speed is None or speed == self._settings.model_speed:
            return
        self._settings = replace(self._settings, model_speed=speed)
        settings_io.save(self._settings)
        self._reload_engine(self._settings.language)

    def _apply_ui_lang(self) -> None:
        """Re-render all static UI text in the active interface language."""
        self.root.title(t("window.title"))
        self._title_label.config(text=t("window.title"))
        self._subtitle_label.config(text=t("status.ready", lang=self._settings.language))
        for menu, idx, key in self._menu_labels:
            menu.entryconfigure(idx, label=t(key))
        # Header labels
        self._lang_label.config(text=t("header.language"))
        self._model_label.config(text=t("header.model"))
        # Speed combo: update values and re-select the translated label for the current speed
        self._speed_combo.config(values=[t("speed.fast"), t("speed.normal")])
        self._speed_var.set(t(f"speed.{self._settings.model_speed}"))
        # Buttons (record label depends on recording state)
        if not self._recording:
            self._record_btn.config(text=t("btn.record"))
        self._clear_btn.config(text=t("btn.clear"))
        self._copy_btn.config(text=t("btn.copy"))
        self._mic_test_btn.config(text=t("btn.mic_test"))

    def _reload_engine(self, lang: str) -> None:
        if self._recording:
            self._stop_recording()
        self._engine_ready = False
        set_button_enabled(self._record_btn, False)
        use_gpu = self._settings.compute_device == "cuda"
        self._status_var.set(loading_status(get_model_size(lang, self._settings.model_speed, use_gpu), lang, use_gpu))
        self.root.config(cursor="watch")
        self._mgr.reload(lang, self._settings.prompts[lang], self._settings.model_speed, self._settings.compute_device)

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
            with self._write_ctx():
                self._text.insert(tk.INSERT, " ")
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

        # Start audio capture for post-recording re-transcription (if enabled).
        if self._settings.asr_postprocess:
            from transcribe_app.session_file_manager import SessionFileManager  # noqa: PLC0415
            self._session_mgr    = SessionFileManager(wav_dir=SESSIONS_DIR, diff_dir=DIFF_DIR)
            self._mgr.audio_sink = self._session_mgr.write_chunk

        set_button_enabled(self._clear_btn, False)
        self._text.config(state=tk.DISABLED)
        self._text.edit_reset()
        self._mgr.start_session()

    def _stop_recording(self) -> None:
        self._recording = False
        self._session_draining = True
        self._record_btn.config(
            text=t("btn.record"),
            bg=C_ACCENT, activebackground=C_ACCENT_H,
            state=tk.DISABLED,
        )
        hoverable(self._record_btn, C_ACCENT, C_ACCENT_H)
        self._status_var.set(t("status.processing"))
        self._text.config(state=tk.NORMAL)
        self._text.edit_reset()
        self._mgr.audio_sink = None
        if self._session_mgr is not None:
            self._session_mgr.finish_recording()

        self._mgr.stop_session()

    # ── UI queue polling ───────────────────────────────────────────────────────

    def _poll_ui(self) -> None:
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                match msg[0]:
                    case "status":
                        if not self._retranscribing:
                            self._status_var.set(msg[1])
                    case "enable_controls":
                        _, ok = msg
                        self._engine_ready = ok
                        if ok:
                            self._set_record_btn_state()
                        # else: keep disabled — status bar already shows the error
                        self.root.config(cursor="")
                    case "update":
                        _, committed, buffer = msg
                        self._set_text(committed, buffer)
                    case "finalise":
                        _, committed = msg
                        self._set_text(committed, "")
                        self._postprocess()  # owns re-enabling the record button
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

    @contextlib.contextmanager
    def _write_ctx(self):
        self._text.config(state=tk.NORMAL)
        try:
            yield
        finally:
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

        with self._write_ctx():
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
        """Apply voice commands to the live transcription immediately after recording ends.

        If ASR post-processing is enabled and WAV files were captured,
        re-transcription runs in a background thread so the UI stays responsive;
        the record button is re-enabled straight away with the live text visible.
        """
        full = self._text.get("1.0", "end-1c")

        # Split off the preserved suffix (text that was after the cursor).
        if self._session_suffix and self._session_suffix in full:
            idx    = full.rfind(self._session_suffix)
            body   = full[:idx]
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

        # Apply voice commands to the live transcription immediately.
        processed      = apply_commands_full(clean(new_text))
        new_text_final = processed if processed is not None else new_text

        if processed is not None:
            with self._write_ctx():
                self._text.delete("1.0", tk.END)
                if prefix:
                    self._text.insert(tk.END, prefix)
                self._text.insert(tk.END, new_text_final)
                if suffix:
                    self._text.insert(tk.END, suffix)
                self._render_markdown()
            self._text.see(tk.END)
            self._session_prefix = prefix + new_text_final

        session_mgr = self._session_mgr

        # If ASR post-processing is enabled and WAV files were captured, run
        # re-transcription in a background thread so the UI stays responsive.
        if session_mgr and session_mgr.wav_paths:
            self._retranscribing = True
            self._status_var.set(t("status.retranscribing"))

            def _run() -> None:
                try:
                    result: "str | None" = self._do_retranscribe(session_mgr, new_text_final)
                except Exception:
                    _log.error("Re-transcription failed", exc_info=True)
                    result = None
                self.root.after(
                    0, lambda: self._on_retranscribe_done(session_mgr, prefix, result, suffix)
                )

            threading.Thread(target=_run, daemon=True).start()
            # _session_draining stays True; _on_retranscribe_done will clear it.
            return

        # No re-transcription — finalise immediately.
        self._session_mgr = None
        self._session_draining = False
        self._schedule_recording_cleanup()
        if not self._finalise_session():
            self._status_var.set(t("status.ready", lang=self._settings.language))

    def _do_retranscribe(self, session_mgr: SessionFileManager, live_text: str) -> "str | None":
        """Background task: re-transcribe captured WAV files via the engine protocol and write a diff."""
        import numpy as np
        from scipy.io import wavfile

        prompt = self._settings.prompts[self._settings.language]

        parts: list[np.ndarray] = []
        for wav_path in session_mgr.wav_paths:
            if not wav_path.exists():
                _log.warning("Re-transcription: WAV file missing: %s", wav_path)
                continue
            sr, chunk = wavfile.read(wav_path)
            if sr != SAMPLE_RATE:
                _log.warning(
                    "Re-transcription: unexpected sample rate %d in %s (expected %d)",
                    sr, wav_path.name, SAMPLE_RATE,
                )
            parts.append(chunk)

        if not parts:
            raise RuntimeError("No audio files available for re-transcription")

        audio = np.concatenate(parts).astype(np.float32) / 32768.0
        retranscribed = self._mgr.transcribe_audio(audio, prompt)

        if retranscribed is None:
            _log.debug("Re-transcription not supported by the active ASR backend — keeping live text")
            return None

        sid = session_mgr.session_id
        diff_lines = list(difflib.unified_diff(
            live_text.splitlines(keepends=True),
            retranscribed.splitlines(keepends=True),
            fromfile=f"{sid}_live",
            tofile=f"{sid}_retranscribed",
        ))
        session_mgr.diff_path.write_text("\n".join(diff_lines), encoding="utf-8")
        _log.info("Re-transcription diff written to %s", session_mgr.diff_path)

        return retranscribed

    def _on_retranscribe_done(
        self,
        session_mgr: "SessionFileManager",
        prefix: str,
        retranscribed: "str | None",
        suffix: str,
    ) -> None:
        """UI-thread callback: update text with re-transcribed content, clean up, re-enable."""
        self._retranscribing = False

        if retranscribed is not None:
            processed = apply_commands_full(clean(retranscribed))
            final     = processed if processed is not None else retranscribed

            with self._write_ctx():
                self._text.delete("1.0", tk.END)
                if prefix:
                    self._text.insert(tk.END, prefix)
                self._text.insert(tk.END, final)
                if suffix:
                    self._text.insert(tk.END, suffix)
                self._render_markdown()
            self._text.see(tk.END)
            self._session_prefix = prefix + final

            diff_exists = session_mgr.diff_path.exists()
            status = (
                t("status.diff_saved", name=session_mgr.diff_path.name)
                if diff_exists
                else t("status.ready", lang=self._settings.language)
            )
            self._status_var.set(status)
        else:
            self._status_var.set(t("status.ready", lang=self._settings.language))

        self._session_mgr = None
        self._session_draining = False
        self._schedule_recording_cleanup()
        self._finalise_session()

    def _schedule_recording_cleanup(self) -> None:
        """Delete stale session/VAD recordings on a background thread when enabled."""
        if self._settings.cleanup_recordings:
            start_async_recordings_cleanup(self._RECORDING_CLEANUP_MAX_AGE_S)

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
        with self._write_ctx():
            self._text.delete("1.0", tk.END)
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
        with self._write_ctx():
            self._text.delete("1.0", tk.END)
            self._text.insert(tk.END, text)
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

    def _open_settings(self) -> None:
        if self._popup_open():
            return
        from .dialogs.settings_dialog import SettingsDialog

        def on_save(new: Settings) -> None:
            old_lang        = self._settings.language
            old_prompt      = self._settings.prompts[old_lang]
            old_speed       = self._settings.model_speed
            old_ui_lang     = self._settings.ui_language
            old_device      = self._settings.compute_device
            old_engine_type = self._settings.engine_type
            old_silence_gap = self._settings.vad_silence_gap
            self._settings = new
            settings_io.save(self._settings)
            self._lang_var.set(new.language)
            self._speed_var.set(t(f"speed.{new.model_speed}"))
            if new.ui_language != old_ui_lang:
                set_language(new.ui_language)
                self._apply_ui_lang()

            # Determine which engine action is needed (if any).
            if new.engine_type != old_engine_type:
                action = self._recreate_manager
            elif (
                new.language != old_lang
                or new.prompts[new.language] != old_prompt
                or new.model_speed != old_speed
                or new.compute_device != old_device
                or new.vad_silence_gap != old_silence_gap
            ):
                if hasattr(self._mgr, "vad_silence_gap"):
                    self._mgr.vad_silence_gap = new.vad_silence_gap
                action = lambda: self._reload_engine(new.language)
            else:
                action = None

            if action is None:
                return

            # Stop any active recording before touching the engine.
            if self._recording:
                self._stop_recording()

            # If a session is still draining, defer until it finishes.
            if self._session_draining:
                self._pending_engine_action = action
            else:
                action()

        dialog = SettingsDialog(self.root, self._settings, on_save)
        self._register_popup(dialog._win)

    def _open_mic_test(self) -> None:
        if self._popup_open():
            return
        from .mic_test import MicTestWindow

        def _on_device_save(new: Settings) -> None:
            self._settings = new
            settings_io.save(self._settings)
            self._mgr.mic_gain = new.mic_gain
            if hasattr(self._mgr, "input_device"):
                self._mgr.input_device = new.input_device

        win = MicTestWindow(self.root, self._settings, _on_device_save)
        self._register_popup(win._win)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._mgr.shutdown()
        self.root.destroy()
