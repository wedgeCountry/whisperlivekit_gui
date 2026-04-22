# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Tkinter GUI application for live speech transcription powered by [WhisperLiveKit](https://github.com/QUENTINFUXA/WHISPERLIVEKIT). Supports English and German. Has two pluggable transcription backends selectable at runtime.

The code lives in the `transcribe_app/` package. The original single-file prototype is preserved as `transcribe_app.py`.

## Running the app

```bash
source .venv/bin/activate
python -m transcribe_app
```

## Dependencies

```bash
pip install -r requirements.txt
# webrtcvad requires Microsoft Visual C++ 14.0+ on Windows
```

## Running tests

```bash
pytest                                          # all tests
pytest tests/test_text_processing.py           # single file
pytest tests/test_text_processing.py::TestClean::test_ellipsis_removed  # single test
```

## Package layout

```
transcribe_app/
├── __main__.py                  # entry point
├── config.py                    # compile-time constants (audio params, language/model map, default prompts)
├── settings.py                  # Settings dataclass + load()/save() JSON persistence
├── i18n.py                      # t(key) UI localisation (en/de); set_language() called once at startup
├── text_processing.py           # pure functions: clean, apply_commands_full, strip_prompt_leak
├── engine_protocol.py           # EngineManagerProtocol ABC + create_engine_manager() factory
├── engine.py                    # AsyncEngine — model lifecycle and sessions (asyncio thread only)
├── engine_manager.py            # EngineManager — thread management + sounddevice stream (WhisperLiveKit)
├── alternative_engine.py        # SpeechToTextEngine — VAD + faster-whisper (owns its own stream)
├── alternative_engine_manager.py# AlternativeEngineManager — wraps SpeechToTextEngine
├── session_file_manager.py      # WAV capture (rotating files) + diff file path
└── ui/
    ├── theme.py                 # C_*/F_* tokens, make_btn(), hoverable(), apply_ttk_style()
    ├── main_window.py           # TranscriptionApp — view + event wiring
    ├── mic_test.py              # MicTestWindow
    └── dialogs/
        └── settings_dialog.py   # full settings dialog (language, engine, prompts, device, etc.)
```

## Architecture

### Thread model
Tkinter main thread owns all widget access. A daemon thread runs an `asyncio` event loop that owns `AsyncEngine` (model + session). A `sounddevice` callback feeds raw PCM via `asyncio.run_coroutine_threadsafe`. A `queue.Queue` bridges results back to Tkinter, polled every 50 ms via `root.after`.

### Dual engine backends
`EngineManagerProtocol` (`engine_protocol.py`) defines the interface both backends implement. `create_engine_manager(engine_type, ...)` returns the right one. The UI only holds an `EngineManagerProtocol` reference — never a concrete class.

| Setting value | Class | Notes |
|---|---|---|
| `"whisperlive"` | `EngineManager` | WhisperLiveKit streaming; partial transcription while speaking |
| `"faster_whisper"` | `AlternativeEngineManager` | VAD-gated faster-whisper; no partial updates; owns its own sounddevice stream |

`AlternativeEngineManager.open_mic_stream()` is a no-op — the engine opens its own stream in `run()`. For `EngineManager`, `open_mic_stream()` must be called from the UI thread (scheduled via `root.after(0, …)`).

### AsyncEngine / EngineManager split
`AsyncEngine` (`engine.py`) has no threading or sounddevice dependency — it runs entirely on the asyncio thread. `EngineManager` (`engine_manager.py`) owns the background thread, the sounddevice stream, and the public UI API (`start`, `reload`, `start_session`, `stop_session`, `open_mic_stream`, `shutdown`).

### TranscriptionApp
Owns `Settings`, one `EngineManagerProtocol` instance, and all display state. Dialogs receive a `Settings` copy and an `on_save(Settings)` callback — they never hold a reference to the app.

### Settings
Immutable dataclass replaced (not mutated) on save via `dataclasses.replace()`. Fields: `language`, `prompts` (per-language Whisper style prompts), `input_device`, `model_speed` (`"fast"`/`"normal"`), `mic_gain`, `ui_language`, `compute_device` (`"cuda"`/`"cpu"`), `asr_postprocess`, `engine_type`, `vad_silence_gap`.

Persisted to `%APPDATA%\transcribe_app\settings.json` (Windows) or `~/.config/transcribe_app/settings.json` (Linux).

### SessionFileManager
Attached as `engine.audio_sink` during recording. Captures raw int16 PCM to rotating 60-second WAV files under `…/transcribe_app/sessions/`. After the session, if `asr_postprocess` is enabled, the WAV is re-transcribed via `whisper_asr` and a unified diff is written to `…/transcribe_app/diff/`.

### i18n
All UI strings go through `t(key, **kwargs)` from `i18n.py`. `set_language("de")` switches the active language (module-level state, safe to read from the asyncio thread under the GIL). Engine status strings produced on the background thread also use `t()`.

### Display model
Committed (finalised) lines accumulate with normal style. The live in-progress buffer is shown in muted italics (`"buffer"` tag) and replaced on each update. After 5 s of silence during recording, `_restructure()` bakes the current text into `_session_prefix` and resets the display counters.

### Lazy imports
`sounddevice`, `AudioProcessor`, `TranscriptionEngine`, and `WhisperLiveKitConfig` are imported inside the background thread to avoid a PortAudio/CUDA hang on window open.

### Windows audio
On Windows, `open_mic_stream()` tries WASAPI exclusive mode first (bypasses APO noise suppression pipeline). Falls back to shared mode if the device is held by another process. The process is also raised to `HIGH_PRIORITY_CLASS` and audio block size is 0.5 s (vs 0.3 s on Linux) to tolerate OS scheduling jitter.

## Voice commands (spoken → markdown)

| Spoken | Output |
|---|---|
| "newline" / "neue Zeile" | `\n` |
| "new paragraph" / "neuer Absatz" | `\n\n` |
| "heading …" / "Überschrift …" (at sentence start) | `\n# …` |
| "Punkt" / "period" | `. ` |
| "Komma" / "comma" | `, ` |
| "Fragezeichen" / "question mark" | `? ` |
| "Ausrufezeichen" / "exclamation mark" | `! ` |
| "Doppelpunkt" / "colon" | `: ` |
| "Semikolon" / "semicolon" | `; ` |
| "Bindestrich" / "hyphen" / "dash" | `-` |
| "erstens" … "zehntens" | `1.` … `10.` |
| "firstly" … "tenth" | `1.` … `10.` |

## Model cache location

WhisperLiveKit models are cached under `~/.cache/huggingface/hub/` in `models--Systran--faster-whisper-*` or `models--mobiuslabsgmbh--faster-whisper-*` directories.
