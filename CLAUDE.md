# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Tkinter GUI application for live speech transcription powered by [WhisperLiveKit](https://github.com/QUENTINFUXA/WHISPERLIVEKIT). Supports English and German, with optional post-processing via Ollama, LanguageTool, or Spylls.

The code lives in the `transcribe_app/` package. The original single-file prototype is preserved as `transcribe_app.py`.

## Running the app

```bash
source .venv/bin/activate
python -m transcribe_app   # new package
python transcribe_app.py   # original single-file (still works)
```

## Dependencies

Install into the local venv:

```bash
pip install -r requirements.txt
```

System packages needed for Spylls spell-checking (Linux):
```bash
sudo apt install hunspell-en-us hunspell-de-de
```

Ollama (optional, for LLM post-processing): must be running at `http://localhost:11434`.

## Package layout

```
transcribe_app/
├── __main__.py              # entry point: python -m transcribe_app
├── config.py                # all compile-time constants (audio, languages, colour tokens, backends)
├── settings.py              # Settings dataclass + load()/save() JSON persistence
├── text_processing.py       # pure functions: clean, apply_commands, strip_prompt_leak, strip_ollama_junk
├── engine.py                # EngineManager — asyncio loop, model lifecycle, audio stream
├── correction/
│   ├── __init__.py          # Corrector protocol + get_corrector() factory
│   ├── ollama.py            # OllamaCorrector (async HTTP)
│   ├── languagetool.py      # LanguageToolCorrector (lazy Java server, call close() on shutdown)
│   └── spylls.py            # SpyllsCorrector (offline Hunspell)
└── ui/
    ├── theme.py             # C_*/F_* tokens, make_btn(), hoverable(), apply_ttk_style()
    ├── main_window.py       # TranscriptionApp — view + wiring only
    ├── mic_test.py          # MicTestWindow
    └── dialogs/
        ├── settings_dialog.py
        ├── voice_style_dialog.py
        └── system_prompt_dialog.py
```

## Architecture

- **Thread model**: Tkinter main thread owns all widget access. A daemon thread runs an `asyncio` event loop that owns `EngineManager` (model + session). A `sounddevice` callback feeds raw PCM via `asyncio.run_coroutine_threadsafe`. A `queue.Queue` bridges results back to Tkinter, polled every 50 ms via `root.after`.
- **`EngineManager`** (`engine.py`): No tkinter dependency. Communicates upward via five injected callbacks (`on_status`, `on_ready`, `on_update`, `on_finalise`, `on_open_mic`). `open_mic_stream()` must be called from the UI thread (scheduled via `root.after(0, …)`) so that `_stream` is only touched on the UI thread.
- **`TranscriptionApp`** (`ui/main_window.py`): Owns `Settings`, `EngineManager`, corrector cache, and all display state. Dialogs receive a `Settings` copy and an `on_save(Settings)` callback — they never hold a reference to the app itself.
- **`Settings`** (`settings.py`): Immutable dataclass replaced (not mutated) on save via `dataclasses.replace()`.
- **Corrector caching**: `LanguageToolCorrector` and `SpyllsCorrector` are cached in `TranscriptionApp._correctors` keyed by `(backend, lang)`. `OllamaCorrector` is created fresh each call so it always reads the current settings (model name, system prompt). Call `corrector.close()` on `LanguageToolCorrector` instances at shutdown to stop the Java server.
- **Lazy imports**: `sounddevice`, `AudioProcessor`, `TranscriptionEngine`, and `WhisperLiveKitConfig` are imported inside the background thread to avoid a PortAudio/CUDA hang on window open.
- **Display model**: Committed (finalised) lines accumulate with normal style. The live in-progress buffer is shown in muted italics (`"buffer"` tag) and replaced on each update. After 5 s of silence during recording, `_restructure()` bakes the current text into `_session_prefix` and resets the display counters.
- **Settings persistence**: `~/.config/transcribe_app/settings.json` (Linux) or `%APPDATA%\transcribe_app\settings.json` (Windows).

## Voice commands (spoken → markdown)

| Spoken | Output |
|---|---|
| "newline" / "neue Zeile" | `\n` |
| "new paragraph" / "neuer Absatz" | `\n\n` |
| "heading …" / "Überschrift …" (at sentence start) | `\n# …` |

## Model cache location

WhisperLiveKit models are cached under `~/.cache/huggingface/hub/` in `models--Systran--faster-whisper-*` or `models--mobiuslabsgmbh--faster-whisper-*` directories.
