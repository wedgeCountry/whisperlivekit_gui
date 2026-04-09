"""
Live transcription Tkinter app using WhisperLiveKit.

Architecture:
  - Main thread: Tkinter UI
  - Background thread: asyncio event loop (TranscriptionEngine + AudioProcessor)
  - sounddevice callback feeds raw PCM chunks via asyncio.run_coroutine_threadsafe
  - threading.Queue bridges async results → Tkinter UI updates (polled every 50 ms)

Display:
  - Committed (finalised) lines accumulate in the text area with normal style.
  - The live in-progress buffer is shown in muted italics and replaced on every
    update until committed.
"""

import asyncio
import json
import math
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

import httpx
import numpy as np

# sounddevice and whisperlivekit are imported lazily to avoid blocking window open
# sounddevice is imported when the mic stream is first opened (PortAudio init can hang)
sd = None  # type: ignore[assignment]
AudioProcessor = None  # type: ignore[assignment]
TranscriptionEngine = None  # type: ignore[assignment]
WhisperLiveKitConfig = None  # type: ignore[assignment]

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHANNELS      = 1
DTYPE         = "int16"
CHUNK_SECONDS = 0.1

# ── GPU detection ──────────────────────────────────────────────────────────────
try:
    import torch
    _GPU = torch.cuda.is_available()
except Exception:
    _GPU = False

# ── Language definitions ───────────────────────────────────────────────────────
LANGUAGE_OPTS = {
    "English": dict(
        model_size="large-v3-turbo" if _GPU else "medium.en",
        fallback_model_size="medium.en",
        lan="en",
    ),
    "Deutsch": dict(
        model_size="large-v3-turbo" if _GPU else "medium",
        fallback_model_size="medium",
        lan="de",
    ),
}


def _is_model_cached(model_size: str) -> bool:
    """Return True if the faster-whisper model is already in the HuggingFace cache.

    WhisperLiveKit may pull from either the Systran or the mobiuslabsgmbh
    organisation depending on the model variant (e.g. large-v3-turbo lives
    under mobiuslabsgmbh, everything else under Systran).
    """
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = [
        hub / f"models--Systran--faster-whisper-{model_size}",
        hub / f"models--mobiuslabsgmbh--faster-whisper-{model_size}",
    ]
    try:
        return any(
            (d / "snapshots").is_dir() and any((d / "snapshots").iterdir())
            for d in candidates
        )
    except OSError:
        return False


def _loading_status(model_size: str, lang: str) -> str:
    device = "GPU" if _GPU else "CPU"
    verb   = "Lade" if _is_model_cached(model_size) else "Lade herunter"
    return f"{verb}: {model_size}  ({lang} · {device})…"
DEFAULT_LANGUAGE = "English"

# Default static prompts per language.
# static_init_prompt is re-injected before every audio chunk, so the style
# hint remains active throughout the whole session.
# A prompt that ends mid-sentence with a comma primes Whisper to treat
# short pauses as continuation rather than sentence boundaries.
DEFAULT_PROMPTS: dict[str, str] = {
    "English": "",
    "Deutsch": (
        "Der Nutzer spricht mit Denkpausen. Wenn er eine Denkpause macht, dann füge ein Leerzeichen ein."
    ),
}


def _build_config(lang: str, prompt: str) -> WhisperLiveKitConfig:
    opts = LANGUAGE_OPTS[lang]
    return WhisperLiveKitConfig(
        pcm_input=True,
        vac=True,
        model_size=opts["model_size"],
        lan=opts["lan"],
        static_init_prompt=prompt if prompt.strip() else None,
    )


# ── Settings persistence ───────────────────────────────────────────────────────
_CONFIG_DIR = (
    Path.home() / "AppData" / "Roaming" / "transcribe_app"
    if sys.platform == "win32"
    else Path.home() / ".config" / "transcribe_app"
)
_SETTINGS_FILE = _CONFIG_DIR / "settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Design tokens ──────────────────────────────────────────────────────────────
C_BG        = "#f8fafc"   # window background
C_SURFACE   = "#ffffff"   # text area / cards
C_HEADER    = "#ffffff"   # toolbar
C_BORDER    = "#e2e8f0"   # hairline borders
C_TEXT      = "#1e293b"   # primary text
C_MUTED     = "#64748b"   # secondary / placeholder text
C_ACCENT    = "#3b82f6"   # blue — idle record button
C_ACCENT_H  = "#2563eb"   # blue hover
C_DANGER    = "#ef4444"   # red — active recording
C_DANGER_H  = "#dc2626"   # red hover
C_BTN       = "#ffffff"   # secondary button face
C_BTN_H     = "#f1f5f9"   # secondary button hover
C_STATUS_BG = "#f1f5f9"   # status bar background
C_BUFFER    = "#94a3b8"   # live buffer text

F_UI    = ("TkDefaultFont", 10)
F_MONO  = ("TkFixedFont",   11)
F_SMALL = ("TkDefaultFont",  9)
F_LABEL = ("TkDefaultFont", 10)


# ── Correction backend ────────────────────────────────────────────────────────
CORRECTION_BACKENDS   = ["Ollama", "LanguageTool", "Spylls"]
DEFAULT_BACKEND       = "Ollama"

# LanguageTool language codes per UI language
LT_LANG_CODES: dict[str, str] = {
    "English": "en-US",
    "Deutsch": "de-DE",
}

# Hunspell dictionary codes for Spylls
HUNSPELL_CODES: dict[str, str] = {
    "English": "en_US",
    "Deutsch": "de_DE",
}

# Common locations where hunspell dictionaries are installed on Linux
HUNSPELL_SEARCH_DIRS = [
    Path("/usr/share/hunspell"),
    Path("/usr/share/myspell/dicts"),
    Path("/usr/share/myspell"),
    Path.home() / ".local" / "share" / "hunspell",
]

# ── Ollama post-processing ─────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
DEFAULT_OLLAMA_MODELS: dict[str, str] = {
    "English": "gemma",
    "Deutsch": "sroecker/sauerkrautlm-7b-hero:latest",
}

DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "English": (
        "You are a precise text editor. The user gives you transcribed speech text "
        "that may contain typos and punctuation errors introduced by automatic speech "
        "recognition. Correct only clear typos and obviously wrong punctuation. "
        "Preserve all markdown formatting (# headings, blank-line paragraphs). "
        "Do not rephrase, add, or remove any content. "
        "Return only the corrected text — no explanations, no surrounding quotes."
    ),
    "Deutsch": (
        "Du bist ein präziser Texteditor. Der Benutzer gibt dir transkribierten Sprachtext, "
        "der Tippfehler und Interpunktionsfehler durch automatische Spracherkennung enthalten kann. "
        "Korrigiere nur eindeutige Tippfehler und offensichtlich falsche Interpunktion. "
        "Behalte alle Markdown-Formatierungen (# Überschriften, Leerzeilen-Absätze) bei. "
        "Formuliere nicht um, füge nichts hinzu und entferne keinen Inhalt. "
        "Gib nur den korrigierten Text zurück – keine Erklärungen, keine Anführungszeichen."
    ),
}


import re as _re

def _clean(text: str) -> str:
    """Remove filler artefacts Whisper inserts at pauses."""
    text = text.replace("…", " ").replace("...", " ")
    text = _re.sub(r"  +", " ", text)
    return text


# ── Voice command → markdown substitutions ────────────────────────────────────
# Commands are matched case-insensitively as whole words.
# "heading" / "Überschrift" are only triggered when the word appears at the
# start of a segment (after sentence-ending punctuation or a newline) to
# reduce accidental matches in natural speech.
_CMDS: list[tuple] = [
    # Consume optional leading space and trailing punctuation + whitespace so
    # Whisper's auto-punctuation ("Newline.") doesn't leave orphaned characters.
    (_re.compile(r"[ \t]*\bnewline[.,]?\s*",          _re.I), "\n"),
    (_re.compile(r"[ \t]*\bneue\s+zeile[.,]?\s*",     _re.I), "\n"),
    (_re.compile(r"[ \t]*\bnew\s+paragraph[.,]?\s*",  _re.I), "\n\n"),
    (_re.compile(r"[ \t]*\bneuer?\s+absatz[.,]?\s*",  _re.I), "\n\n"),
    # heading: only fires after ^, newline, or sentence-end punctuation
    (_re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)heading[.,]?\s+",     _re.I | _re.M), "\n# "),
    (_re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)überschrift[.,]?\s+", _re.I | _re.M), "\n# "),
]


def _apply_commands(text: str) -> str:
    for pattern, replacement in _CMDS:
        text = pattern.sub(replacement, text)
    # Collapse accidental triple newlines from stacked commands
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.lstrip("\n")


def _hoverable(btn: tk.Button, normal: str, hover: str) -> None:
    """Attach Enter/Leave hover colour to a tk.Button."""
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda _e: btn.config(bg=normal))


def _make_btn(
    parent: tk.Widget,
    text: str,
    command,
    *,
    primary: bool = False,
    danger: bool = False,
) -> tk.Button:
    if primary:
        bg, fg, hov = C_ACCENT, "#ffffff", C_ACCENT_H
    elif danger:
        bg, fg, hov = C_DANGER, "#ffffff", C_DANGER_H
    else:
        bg, fg, hov = C_BTN, C_TEXT, C_BTN_H

    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg,
        activebackground=hov, activeforeground=fg,
        font=F_UI, relief=tk.FLAT, bd=0,
        padx=20, pady=9, cursor="hand2",
        highlightbackground=C_BORDER,
        highlightthickness=1,
    )
    _hoverable(btn, bg, hov)
    return btn


# ── App ────────────────────────────────────────────────────────────────────────
class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live Transcription")
        self.root.minsize(560, 460)
        self.root.configure(bg=C_BG)

        self._ui_queue: queue.Queue = queue.Queue()
        self._recording  = False
        self._stream: sd.InputStream | None = None

        _s = _load_settings()
        self._current_lang: str = (
            _s.get("language", DEFAULT_LANGUAGE)
            if _s.get("language") in LANGUAGE_OPTS else DEFAULT_LANGUAGE
        )
        self._prompts: dict[str, str] = {
            lang: _s.get("prompts", {}).get(lang, DEFAULT_PROMPTS[lang])
            for lang in LANGUAGE_OPTS
        }
        self._system_prompts: dict[str, str] = {
            lang: _s.get("system_prompts", {}).get(lang, DEFAULT_SYSTEM_PROMPTS[lang])
            for lang in LANGUAGE_OPTS
        }
        self._correction_backend: str = _s.get("correction_backend", DEFAULT_BACKEND)
        self._ollama_models: dict[str, str] = {
            lang: _s.get("ollama_models", {}).get(lang, DEFAULT_OLLAMA_MODELS[lang])
            for lang in LANGUAGE_OPTS
        }
        self._lt_cache: dict = {}   # lang_code → LanguageTool instance

        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine: TranscriptionEngine | None     = None
        self._processor: AudioProcessor | None       = None
        self._session_prefix: str    = ""
        self._last_raw_committed: str = ""   # raw WhisperLiveKit output this session
        self._absorbed_committed: str = ""   # raw text baked into prefix at last restructure
        self._last_text_time: float   = 0.0  # monotonic time text last visibly changed
        self._last_display_sig: str   = ""   # signature of last displayed content

        self._apply_ttk_style()
        self._build_ui()
        self._start_bg_loop()
        self._poll_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── ttk theme ─────────────────────────────────────────────────────────────

    def _apply_ttk_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=C_SURFACE,
            background=C_SURFACE,
            foreground=C_TEXT,
            bordercolor=C_BORDER,
            lightcolor=C_BORDER,
            darkcolor=C_BORDER,
            arrowcolor=C_MUTED,
            arrowsize=14,
            padding=(6, 4),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", C_SURFACE)],
            foreground=[("readonly", C_TEXT)],
            bordercolor=[("focus", C_ACCENT)],
        )

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)   # text area row

        # ── Menu bar ───────────────────────────────────────────────────────────
        menubar = tk.Menu(self.root, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Speichern…", command=self._save_file)
        file_menu.add_command(label="Laden…",     command=self._load_file)

        edit_menu = tk.Menu(menubar, bg=C_SURFACE, fg=C_TEXT, tearoff=0)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        edit_menu.add_command(label="Stimme & Style Prompt…", command=self._open_voice_style)
        edit_menu.add_command(label="System Prompt…",         command=self._open_system_prompt)
        edit_menu.add_separator()
        edit_menu.add_command(label="Einstellungen…",         command=self._open_settings)

        # ── Row 0: header ─────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=C_HEADER)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        tk.Frame(self.root, bg=C_BORDER, height=1).grid(
            row=0, column=0, sticky="sew"
        )

        tk.Label(
            header, text="Live Transcription",
            bg=C_HEADER, fg=C_TEXT,
            font=("TkDefaultFont", 13, "bold"),
            padx=16, pady=12,
        ).grid(row=0, column=0, sticky="w")

        # language selector (right-aligned in toolbar)
        lang_frame = tk.Frame(header, bg=C_HEADER)
        lang_frame.grid(row=0, column=2, padx=12, pady=8, sticky="e")

        tk.Label(
            lang_frame, text="Sprache",
            bg=C_HEADER, fg=C_MUTED, font=F_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self._lang_var = tk.StringVar(value=self._current_lang)
        self._lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly",
            width=10,
            font=F_LABEL,
        )
        self._lang_combo.pack(side=tk.LEFT)
        self._lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        # ── Row 1: text area ───────────────────────────────────────────────────
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(row=1, column=0, sticky="new")
        border_frame = tk.Frame(self.root, bg=C_BORDER)
        border_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        border_frame.rowconfigure(0, weight=1)
        border_frame.columnconfigure(0, weight=1)

        self._text = scrolledtext.ScrolledText(
            border_frame,
            wrap=tk.WORD,
            font=F_MONO,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            selectbackground=C_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT, bd=0,
            padx=14, pady=12,
        )
        self._text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._text.tag_configure(
            "buffer",
            foreground=C_BUFFER,
            font=("TkFixedFont", 11, "italic"),
        )
        self._text.tag_configure(
            "h1",
            font=("TkDefaultFont", 17, "bold"),
            foreground=C_ACCENT,
            spacing1=10, spacing3=4,
        )
        self._text.tag_configure(
            "h2",
            font=("TkDefaultFont", 13, "bold"),
            foreground=C_TEXT,
            spacing1=8, spacing3=2,
        )
        self._text.tag_configure(
            "h3",
            font=("TkDefaultFont", 11, "bold"),
            foreground=C_MUTED,
            spacing1=4,
        )
        self._text.tag_configure(
            "para_space",
            spacing1=10,
        )

        # ── Row 2: button bar ──────────────────────────────────────────────────
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(
            row=2, column=0, sticky="ew"
        )
        btn_bar = tk.Frame(self.root, bg=C_BG, pady=10)
        btn_bar.grid(row=2, column=0, sticky="ew")
        btn_bar.columnconfigure((0, 1, 2, 3, 4), weight=1)

        self._record_btn = _make_btn(
            btn_bar, "⏺  Record", self._toggle_recording, primary=True
        )
        self._record_btn.config(state=tk.DISABLED)
        self._record_btn.grid(row=0, column=0, padx=(12, 6))

        _make_btn(btn_bar, "Clear",    self._clear_text   ).grid(row=0, column=1, padx=6)
        _make_btn(btn_bar, "Copy",     self._copy_text    ).grid(row=0, column=2, padx=6)

        self._polish_btn = _make_btn(btn_bar, "Rechtschreibkorrektur", self._postprocess_text)
        self._polish_btn.grid(row=0, column=3, padx=6)

        _make_btn(btn_bar, "Test Mic", self._open_mic_test).grid(row=0, column=4, padx=(6, 12))

        # ── Row 3 / 4: status bar ─────────────────────────────────────────────
        tk.Frame(self.root, bg=C_BORDER, height=1).grid(
            row=3, column=0, sticky="ew"
        )
        self._status_var = tk.StringVar(
            value=_loading_status(
                LANGUAGE_OPTS[self._current_lang]["model_size"],
                self._current_lang,
            )
        )
        tk.Label(
            self.root, textvariable=self._status_var,
            anchor="w", padx=12, pady=5,
            bg=C_STATUS_BG, fg=C_MUTED, font=F_SMALL,
        ).grid(row=4, column=0, sticky="ew")

        self.root.config(cursor="watch")

    # ── Background asyncio thread ──────────────────────────────────────────────

    def _start_bg_loop(self) -> None:
        def _run() -> None:
            global AudioProcessor, TranscriptionEngine, WhisperLiveKitConfig
            from whisperlivekit import AudioProcessor, TranscriptionEngine  # noqa: F811
            from whisperlivekit.config import WhisperLiveKitConfig  # noqa: F811

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._load_engine(self._current_lang))
            self._loop.run_forever()

        threading.Thread(target=_run, daemon=True).start()

    async def _load_engine(self, lang: str) -> None:
        opts         = LANGUAGE_OPTS[lang]
        model_size   = opts["model_size"]
        fallback     = opts["fallback_model_size"]
        prompt       = self._prompts[lang]

        TranscriptionEngine.reset()
        self._ui_queue.put(("status", _loading_status(model_size, lang)))
        config = _build_config(lang, prompt)
        try:
            self._engine = TranscriptionEngine(config=config)
        except Exception as exc:
            if _GPU and model_size != fallback:
                # GPU model failed (e.g. OOM) — retry with the smaller CPU model
                self._ui_queue.put((
                    "status",
                    f"Fehler beim Laden ({exc.__class__.__name__}), "
                    f"Fallback: {_loading_status(fallback, lang)}",
                ))
                TranscriptionEngine.reset()
                fallback_cfg = WhisperLiveKitConfig(
                    pcm_input=True, vac=True,
                    model_size=fallback,
                    lan=opts["lan"],
                    static_init_prompt=prompt if prompt.strip() else None,
                )
                self._engine = TranscriptionEngine(config=fallback_cfg)
            else:
                self._ui_queue.put(("status", f"Fehler: {exc}"))
                self._ui_queue.put(("enable_controls",))
                return
        self._ui_queue.put(("status", f"Bereit  ·  {lang} Modell geladen"))
        self._ui_queue.put(("enable_controls",))

    # ── Language selection ─────────────────────────────────────────────────────

    def _on_language_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        if lang == self._current_lang:
            return
        self._current_lang = lang
        self._save_all_settings()
        self._reload_engine_ui(lang)

    def _open_voice_style(self) -> None:
        VoiceStyleDialog(self.root, self)

    def _save_all_settings(self) -> None:
        _save_settings({
            "language":           self._current_lang,
            "prompts":            self._prompts,
            "system_prompts":     self._system_prompts,
            "ollama_models":      self._ollama_models,
            "correction_backend": self._correction_backend,
        })

    def _reload_engine_ui(self, lang: str) -> None:
        if self._recording:
            self._stop_recording()
        self._record_btn.config(state=tk.DISABLED)
        self._status_var.set(_loading_status(LANGUAGE_OPTS[lang]["model_size"], lang))
        self.root.config(cursor="watch")
        asyncio.run_coroutine_threadsafe(self._load_engine(lang), self._loop)

    # ── Recording session ──────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._recording = True
        self._record_btn.config(
            text="⏹  Stop",
            bg=C_DANGER, activebackground=C_DANGER_H,
        )
        _hoverable(self._record_btn, C_DANGER, C_DANGER_H)
        self._status_var.set("Recording…")
        existing = self._text.get("1.0", tk.END).rstrip("\n")
        self._session_prefix    = (existing + "\n") if existing else ""
        self._last_raw_committed = ""
        self._absorbed_committed = ""
        self._last_text_time     = time.monotonic()  # don't fire immediately
        self._last_display_sig   = ""
        asyncio.run_coroutine_threadsafe(self._run_session(), self._loop)

    def _stop_recording(self) -> None:
        self._recording = False
        self._record_btn.config(
            text="⏺  Record",
            bg=C_ACCENT, activebackground=C_ACCENT_H,
        )
        _hoverable(self._record_btn, C_ACCENT, C_ACCENT_H)
        self._status_var.set("Processing remaining audio…")
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._processor is not None:
            asyncio.run_coroutine_threadsafe(
                self._processor.process_audio(None), self._loop
            )

    async def _run_session(self) -> None:
        self._processor = AudioProcessor(transcription_engine=self._engine)
        results_gen = await self._processor.create_tasks()
        self.root.after(0, self._open_mic_stream)

        committed_text = ""
        async for front_data in results_gen:
            lines_text = " ".join(
                seg.text for seg in front_data.lines if seg.text
            )
            if lines_text and lines_text != committed_text:
                committed_text = lines_text
            buffer = front_data.buffer_transcription or ""
            self._ui_queue.put(("update", committed_text, buffer))

        self._ui_queue.put(("finalise", committed_text))
        self._ui_queue.put(("status", f"Ready  ·  {self._current_lang} model loaded"))

    def _open_mic_stream(self) -> None:
        global sd
        import sounddevice as sd  # noqa: F811

        loop      = self._loop
        processor = self._processor
        blocksize = int(SAMPLE_RATE * CHUNK_SECONDS)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if not self._recording:
                return
            asyncio.run_coroutine_threadsafe(
                processor.process_audio(indata.tobytes()), loop
            )

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=blocksize, callback=_callback,
        )
        self._stream.start()

    # ── Text area helpers ──────────────────────────────────────────────────────

    def _strip_prompt_leak(self, text: str) -> str:
        """Remove the Whisper static_init_prompt if it leaks into transcribed text.

        Whisper occasionally hallucinates the init-prompt as spoken content,
        especially when it is re-injected before every audio chunk.
        """
        prompt = self._prompts[self._current_lang].strip()
        if not prompt or not text:
            return text
        # Case-insensitive removal of every occurrence.
        text = _re.sub(_re.escape(prompt), " ", text, flags=_re.IGNORECASE)
        return _re.sub(r"  +", " ", text).strip()

    # Regex patterns for common LLM preamble/postamble in both languages
    _OLLAMA_JUNK = _re.compile(
        r"^(?:"
        # English preambles
        r"here\s+is\s+(?:the\s+)?(?:corrected\s+)?(?:text|version)[^:]*:\s*|"
        r"(?:the\s+)?(?:corrected|revised|edited)\s+(?:text|version)[^:]*:\s*|"
        r"(?:i\s+have\s+)?(?:corrected|revised|fixed)[^:\n]*:\s*|"
        r"(?:sure|certainly|of\s+course)[^:\n]*[:.]\s*|"
        # German preambles
        r"hier\s+ist\s+(?:der\s+)?(?:korrigierte\s+)?(?:text|version)[^:]*:\s*|"
        r"(?:der\s+)?(?:korrigierte|bearbeitete|überarbeitete)\s+(?:text|version)[^:]*:\s*|"
        r"ich\s+habe[^:\n]*(?:korrigiert|bearbeitet|überarbeitet)[^:\n]*[:.]\s*|"
        r"(?:sicher|natürlich|gerne)[^:\n]*[:.]\s*"
        r")+",
        _re.IGNORECASE | _re.DOTALL,
    )

    def _strip_ollama_leak(self, text: str) -> str:
        """Remove system-prompt echo and common LLM preambles from Ollama output."""
        # Strip echoed system prompt
        sys_prompt = self._system_prompts[self._current_lang].strip()
        if sys_prompt:
            text = _re.sub(_re.escape(sys_prompt), "", text, flags=_re.IGNORECASE).strip()
        # Strip common preamble phrases
        text = self._OLLAMA_JUNK.sub("", text).strip()
        # Strip surrounding markdown code fences the model sometimes adds
        text = _re.sub(r"^```[^\n]*\n?", "", text).strip()
        text = _re.sub(r"\n?```$",        "", text).strip()
        return text

    def _set_text(self, committed: str, buffer: str) -> None:
        # Track raw committed for restructure deduplication.
        self._last_raw_committed = committed

        # If a restructure already absorbed part of the committed history,
        # strip that prefix so we don't repeat it after _session_prefix.
        display_committed = committed
        if self._absorbed_committed:
            if committed.startswith(self._absorbed_committed):
                display_committed = committed[len(self._absorbed_committed):]
            else:
                # Whisper revised the old text — reset the absorbed marker.
                self._absorbed_committed = ""

        display_committed = _apply_commands(_clean(self._strip_prompt_leak(display_committed)))
        display_buffer    = _clean(self._strip_prompt_leak(buffer))

        # Reset the silence timer only when the visible content changes.
        sig = display_committed + "\x00" + display_buffer
        if sig != self._last_display_sig:
            self._last_display_sig = sig
            self._last_text_time   = time.monotonic()

        self._text.delete("1.0", tk.END)
        if self._session_prefix:
            self._text.insert(tk.END, self._session_prefix)
        if display_committed:
            self._text.insert(tk.END, display_committed)
        if display_buffer:
            self._text.insert(tk.END, (" " if display_committed else "") + display_buffer, "buffer")
        self._render_markdown()
        self._text.see(tk.END)

    def _restructure(self) -> None:
        """Apply _clean + _apply_commands to the full text area after a silence pause."""
        current = self._text.get("1.0", tk.END).rstrip()
        if not current:
            self._last_text_time = 0.0
            return
        restructured = _apply_commands(_clean(current))
        self._session_prefix     = restructured + "\n"
        self._absorbed_committed = self._last_raw_committed
        self._last_text_time     = 0.0   # don't fire again until new text arrives
        self._last_display_sig   = ""

        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, self._session_prefix)
        self._render_markdown()
        self._text.see(tk.END)

    def _render_markdown(self) -> None:
        """Apply visual heading tags based on markdown syntax in the text widget."""
        for tag in ("h1", "h2", "h3", "para_space"):
            self._text.tag_remove(tag, "1.0", tk.END)

        content = self._text.get("1.0", tk.END)
        for i, line in enumerate(content.split("\n"), start=1):
            if line.startswith("### "):
                self._text.tag_add("h3", f"{i}.0", f"{i}.end")
            elif line.startswith("## "):
                self._text.tag_add("h2", f"{i}.0", f"{i}.end")
            elif line.startswith("# "):
                self._text.tag_add("h1", f"{i}.0", f"{i}.end")
            elif line.strip() == "":
                self._text.tag_add("para_space", f"{i}.0", f"{i}.end")

    def _clear_text(self) -> None:
        self._text.delete("1.0", tk.END)

    def _copy_text(self) -> None:
        content = self._text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._status_var.set("Copied to clipboard ✓")
        self.root.after(2000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._current_lang} model loaded"
        ))

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
                    case "polish_done":
                        _, corrected = msg
                        self._text.delete("1.0", tk.END)
                        self._text.insert(tk.END, corrected)
                        self._session_prefix     = corrected + "\n"
                        self._last_raw_committed  = ""
                        self._absorbed_committed  = ""
                        self._last_display_sig    = ""
                        self._last_text_time      = 0.0
                        self._render_markdown()
                        self._polish_btn.config(state=tk.NORMAL)
                        self.root.config(cursor="")
                        self._status_var.set(
                            f"Ready  ·  {self._current_lang} model loaded"
                        )
                    case "polish_error":
                        _, err = msg
                        self._status_var.set(f"Rechtschreibkorrektur fehlgeschlagen: {err}")
                        self._polish_btn.config(state=tk.NORMAL)
                        self.root.config(cursor="")
        except queue.Empty:
            pass

        # Auto-restructure after 5 seconds of silence while recording.
        if (
            self._recording
            and self._last_text_time > 0.0
            and (time.monotonic() - self._last_text_time) > 5.0
        ):
            self._restructure()

        self.root.after(50, self._poll_ui)

    # ── File menu actions ──────────────────────────────────────────────────────

    def _save_file(self) -> None:
        text = self._text.get("1.0", tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Textdatei", "*.txt"),
                ("Markdown",  "*.md"),
                ("Alle Dateien", "*"),
            ],
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")
        name = Path(path).name
        self._status_var.set(f"Gespeichert: {name}")
        self.root.after(3000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._current_lang} model loaded"
        ))

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Textdatei", "*.txt"),
                ("Markdown",  "*.md"),
                ("Alle Dateien", "*"),
            ],
        )
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8")
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, text)
        self._session_prefix     = text
        self._last_raw_committed  = ""
        self._absorbed_committed  = ""
        self._last_display_sig    = ""
        self._last_text_time      = 0.0
        self._render_markdown()
        name = Path(path).name
        self._status_var.set(f"Geladen: {name}")
        self.root.after(3000, lambda: self._status_var.set(
            "Recording…" if self._recording
            else f"Ready  ·  {self._current_lang} model loaded"
        ))

    # ── Edit menu actions ──────────────────────────────────────────────────────

    def _open_system_prompt(self) -> None:
        SystemPromptDialog(self.root, self)

    def _open_settings(self) -> None:
        SettingsDialog(self.root, self)

    # ── Rechtschreibkorrektur ──────────────────────────────────────────────────

    def _postprocess_text(self) -> None:
        if self._recording:
            return
        text = self._text.get("1.0", tk.END).strip()
        if not text:
            return

        self._polish_btn.config(state=tk.DISABLED)
        self._status_var.set(
            f"Rechtschreibkorrektur läuft  ({self._correction_backend})…"
        )
        self.root.config(cursor="watch")

        async def _task() -> None:
            try:
                result = await self._run_correction(text)
                self._ui_queue.put(("polish_done", result))
            except Exception as exc:
                self._ui_queue.put(("polish_error", str(exc)))

        asyncio.run_coroutine_threadsafe(_task(), self._loop)

    async def _run_correction(self, text: str) -> str:
        """Dispatch to the selected correction backend."""
        if self._correction_backend == "LanguageTool":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._run_languagetool, text)
        if self._correction_backend == "Spylls":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._run_spylls, text)
        # Default: Ollama
        return await self._run_ollama(text)

    async def _run_ollama(self, text: str) -> str:
        payload = {
            "model": self._ollama_models[self._current_lang],
            "messages": [
                {"role": "system", "content": self._system_prompts[self._current_lang]},
                {"role": "user",   "content": text},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            )
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            return self._strip_ollama_leak(result)

    def _run_languagetool(self, text: str) -> str:
        try:
            import language_tool_python  # type: ignore
        except ImportError:
            raise RuntimeError(
                "LanguageTool nicht installiert.\n"
                "Bitte ausführen: pip install language_tool_python"
            )
        lang_code = LT_LANG_CODES[self._current_lang]
        if lang_code not in self._lt_cache:
            self._lt_cache[lang_code] = language_tool_python.LanguageTool(lang_code)
        tool    = self._lt_cache[lang_code]
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def _run_spylls(self, text: str) -> str:
        try:
            from spylls.hunspell import Dictionary  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Spylls nicht installiert.\n"
                "Bitte ausführen: pip install spylls"
            )
        code = HUNSPELL_CODES[self._current_lang]
        dic  = self._find_hunspell_dict(code)
        if dic is None:
            raise RuntimeError(
                f"Hunspell-Wörterbuch '{code}' nicht gefunden.\n"
                f"Bitte installieren: sudo apt install hunspell-{code[:2].lower()}"
            )
        d = Dictionary.from_files(str(dic))

        def _fix(m: _re.Match) -> str:
            word = m.group(0)
            if d.lookup(word):
                return word
            suggestions = list(d.suggest(word))
            return suggestions[0] if suggestions else word

        return _re.sub(r"[^\W\d_]+", _fix, text)

    @staticmethod
    def _find_hunspell_dict(code: str) -> Path | None:
        for base in HUNSPELL_SEARCH_DIRS:
            if (base / f"{code}.dic").exists():
                return base / code
        return None

    def _open_mic_test(self) -> None:
        MicTestWindow(self.root)

    # ── Shutdown / cleanup ─────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Release GPU memory and stop background threads before closing."""
        # Stop audio capture first so no more callbacks fire.
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False

        # Close any cached LanguageTool instances (shuts down the Java server).
        for lt in self._lt_cache.values():
            try:
                lt.close()
            except Exception:
                pass
        self._lt_cache.clear()

        # Drop Python references to the model so the GC can free VRAM.
        self._processor = None
        self._engine = None
        TranscriptionEngine.reset()
        if _GPU:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Stop the event loop (lets the background thread exit cleanly).
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        self.root.destroy()


# ── Mic test window ────────────────────────────────────────────────────────────

class MicTestWindow:
    BAR_W   = 420
    BAR_H   = 32
    POLL_MS = 40
    CHUNK   = 1024

    def __init__(self, parent: tk.Tk) -> None:
        self._stream: sd.InputStream | None = None
        self._amplitude  = 0.0
        self._peak_x     = 0.0
        self._peak_decay = 0.0

        self._win = tk.Toplevel(parent)
        self._win.title("Microphone Test")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._start_stream()
        self._animate()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            outer, text="Microphone input level",
            bg=C_BG, fg=C_TEXT, font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))

        # Bar canvas with border frame
        bar_border = tk.Frame(outer, bg=C_BORDER)
        bar_border.pack(fill=tk.X, pady=(0, 8))

        self._canvas = tk.Canvas(
            bar_border,
            width=self.BAR_W, height=self.BAR_H,
            bg=C_SURFACE, highlightthickness=0,
        )
        self._canvas.pack(padx=1, pady=1)

        # Filled bar
        self._canvas.create_rectangle(
            0, 0, 0, self.BAR_H, fill=C_ACCENT, outline="", tags="bar",
        )
        # Peak tick
        self._canvas.create_rectangle(
            0, 0, 0, self.BAR_H, fill=C_ACCENT_H, outline="", tags="peak",
        )

        # dB readout
        self._db_var = tk.StringVar(value="–∞ dBFS")
        tk.Label(
            outer, textvariable=self._db_var,
            bg=C_BG, fg=C_TEXT, font=("TkFixedFont", 12),
            anchor="center",
        ).pack(fill=tk.X, pady=(0, 8))

        # Device info
        try:
            global sd
            import sounddevice as sd  # noqa: F811
            dev  = sd.query_devices(kind="input")
            info = f"{dev['name']}  ·  {int(dev['default_samplerate'])} Hz"
        except Exception:
            info = "Default input device"
        tk.Label(
            outer, text=info,
            bg=C_BG, fg=C_MUTED, font=F_SMALL,
            anchor="center", wraplength=self.BAR_W,
        ).pack(fill=tk.X)

    def _start_stream(self) -> None:
        global sd
        import sounddevice as sd  # noqa: F811

        def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            self._amplitude = min(rms / 32768.0, 1.0)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=self.CHUNK, callback=_cb,
        )
        self._stream.start()

    def _animate(self) -> None:
        if not self._win.winfo_exists():
            return

        amp = self._amplitude

        if amp >= self._peak_x:
            self._peak_x    = amp
            self._peak_decay = 0.0
        else:
            self._peak_decay += self.POLL_MS / 1000.0
            self._peak_x = max(self._peak_x - self._peak_decay * 0.4, amp)

        bar_px  = int(amp          * self.BAR_W)
        peak_px = int(self._peak_x * self.BAR_W)

        # green → yellow → red
        if amp < 0.5:
            r = int(amp * 2 * 210)
            g = 180
        else:
            r = 210
            g = int((1.0 - (amp - 0.5) * 2) * 180)
        colour = f"#{r:02x}{g:02x}50"

        self._canvas.itemconfig("bar",  fill=colour)
        self._canvas.coords("bar",  0, 2, max(bar_px, 0),  self.BAR_H - 2)
        self._canvas.coords("peak", peak_px - 2, 0, peak_px + 2, self.BAR_H)

        self._db_var.set(
            f"{20 * math.log10(amp):+.1f} dBFS" if amp > 0 else "–∞ dBFS"
        )
        self._win.after(self.POLL_MS, self._animate)

    def _on_close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._win.destroy()


# ── Settings dialog ───────────────────────────────────────────────────────────

class SettingsDialog:
    """Global application settings (currently: correction backend)."""

    def __init__(self, parent: tk.Tk, app: "TranscriptionApp") -> None:
        self._app = app

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

        self._backend_var = tk.StringVar(value=self._app._correction_backend)
        ttk.Combobox(
            outer,
            textvariable=self._backend_var,
            values=CORRECTION_BACKENDS,
            state="readonly",
            width=16,
            font=F_LABEL,
        ).grid(row=0, column=1, sticky="w", pady=(0, 6))

        # Brief description of the selected backend
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
        _make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

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

    def _update_desc(self, *_) -> None:
        self._desc_var.set(self._DESCRIPTIONS.get(self._backend_var.get(), ""))

    def _save(self) -> None:
        self._app._correction_backend = self._backend_var.get()
        self._app._save_all_settings()
        self._win.destroy()


# ── Voice & Style Prompt dialog ───────────────────────────────────────────────

class VoiceStyleDialog:
    """Select voice/language and edit the per-language style prompt."""

    def __init__(self, parent: tk.Tk, app: "TranscriptionApp") -> None:
        self._app = app

        self._win = tk.Toplevel(parent)
        self._win.title("Stimme & Style Prompt")
        self._win.resizable(False, False)
        self._win.configure(bg=C_BG)
        self._win.grab_set()

        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        # Language / voice row
        tk.Label(outer, text="Stimme:", bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._lang_var = tk.StringVar(value=self._app._current_lang)
        ttk.Combobox(
            outer,
            textvariable=self._lang_var,
            values=list(LANGUAGE_OPTS.keys()),
            state="readonly",
            width=14,
            font=F_LABEL,
        ).grid(row=0, column=1, sticky="w", pady=(0, 10))

        # Ollama model row
        tk.Label(outer, text="Ollama-Modell:", bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 10)
        )
        self._model_var = tk.StringVar(
            value=self._app._ollama_models[self._app._current_lang]
        )
        tk.Entry(
            outer,
            textvariable=self._model_var,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, font=F_SMALL,
            highlightbackground=C_BORDER, highlightthickness=1,
            width=20,
        ).grid(row=1, column=1, sticky="w", ipady=4, pady=(0, 10))

        # Style prompt row
        tk.Label(outer, text="Style Prompt:", bg=C_BG, fg=C_MUTED, font=F_SMALL).grid(
            row=2, column=0, sticky="nw", padx=(0, 10), pady=(0, 12)
        )
        border = tk.Frame(outer, bg=C_BORDER)
        border.grid(row=2, column=1, sticky="ew", pady=(0, 12))

        self._prompt_text = tk.Text(
            border,
            wrap=tk.WORD,
            height=4,
            font=F_SMALL,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, bd=0,
            padx=8, pady=6,
        )
        self._prompt_text.pack(padx=1, pady=1)
        self._prompt_text.insert(tk.END, self._app._prompts[self._app._current_lang])

        # Update fields when language changes
        self._lang_var.trace_add("write", self._on_lang_change)

        # Buttons
        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e")
        _make_btn(btn_row, "Zurücksetzen", self._reset).pack(side=tk.LEFT, padx=(0, 8))
        _make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

    def _on_lang_change(self, *_) -> None:
        lang = self._lang_var.get()
        self._model_var.set(self._app._ollama_models[lang])
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, self._app._prompts[lang])

    def _reset(self) -> None:
        lang = self._lang_var.get()
        self._model_var.set(DEFAULT_OLLAMA_MODELS[lang])
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, DEFAULT_PROMPTS[lang])

    def _save(self) -> None:
        lang   = self._lang_var.get()
        prompt = self._prompt_text.get("1.0", tk.END).strip()
        model  = self._model_var.get().strip() or DEFAULT_OLLAMA_MODELS[lang]
        self._app._prompts[lang]       = prompt
        self._app._ollama_models[lang] = model
        self._app._current_lang        = lang
        # Keep the header dropdown in sync
        self._app._lang_var.set(lang)
        self._app._save_all_settings()
        self._win.destroy()
        # Reload engine whenever language or prompt changed
        self._app._reload_engine_ui(lang)


# ── System Prompt dialog ───────────────────────────────────────────────────────

class SystemPromptDialog:
    """Editable system prompt for the Rechtschreibkorrektur feature, per language."""

    def __init__(self, parent: tk.Tk, app: "TranscriptionApp") -> None:
        self._app = app

        self._win = tk.Toplevel(parent)
        self._win.title("System Prompt – Rechtschreibkorrektur")
        self._win.minsize(500, 340)
        self._win.configure(bg=C_BG)
        self._win.grab_set()   # modal

        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self._win, bg=C_BG, padx=16, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # Language selector
        top = tk.Frame(outer, bg=C_BG)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(top, text="Sprache:", bg=C_BG, fg=C_MUTED, font=F_SMALL).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self._lang_var = tk.StringVar(value=self._app._current_lang)
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

        # Text area for the prompt
        border = tk.Frame(outer, bg=C_BORDER)
        border.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        border.rowconfigure(0, weight=1)
        border.columnconfigure(0, weight=1)

        self._prompt_text = tk.Text(
            border,
            wrap=tk.WORD,
            font=F_SMALL,
            bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, bd=0,
            padx=10, pady=8,
        )
        self._prompt_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._prompt_text.insert(tk.END, self._app._system_prompts[self._app._current_lang])

        # Buttons
        btn_row = tk.Frame(outer, bg=C_BG)
        btn_row.grid(row=2, column=0, sticky="e")

        _make_btn(btn_row, "Zurücksetzen", self._reset).pack(side=tk.LEFT, padx=(0, 8))
        _make_btn(btn_row, "Speichern", self._save, primary=True).pack(side=tk.LEFT)

    def _on_lang_change(self, _event=None) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, self._app._system_prompts[lang])

    def _reset(self) -> None:
        lang = self._lang_var.get()
        self._prompt_text.delete("1.0", tk.END)
        self._prompt_text.insert(tk.END, DEFAULT_SYSTEM_PROMPTS[lang])

    def _save(self) -> None:
        lang = self._lang_var.get()
        self._app._system_prompts[lang] = self._prompt_text.get("1.0", tk.END).strip()
        self._app._save_all_settings()
        self._win.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
