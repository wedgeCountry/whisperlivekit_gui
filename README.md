# WhisperLiveKit GUI

A desktop application for real-time speech transcription, built with Tkinter and powered by [WhisperLiveKit](https://github.com/QUENTINFUXA/WHISPERLIVEKIT) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Implemented using [Claude Code](https://claude.ai/code).

## Features

- **Live transcription** — continuous speech-to-text with low latency
- **Two engine backends** — WhisperLive (streaming) or Faster Whisper (VAD-gated)
- **English and German** — with per-language Whisper style prompts for improved accuracy
- **GPU acceleration** — automatically uses CUDA when available; falls back to CPU
- **Voice commands** — dictate punctuation, line breaks, headings, and numbered lists by speaking them aloud
- **ASR post-processing** — optional re-transcription of the full session audio after recording ends, with a saved diff
- **Bilingual UI** — interface available in English and German
- **Microphone test** — built-in level meter to verify audio input before recording
- **Session persistence** — save and load transcripts as plain text files
- **Adjustable model speed** — Fast (smaller model) or Normal (larger model) per language
- **Mic gain control** — software amplification applied before the ASR model
- **Settings persistence** — all preferences saved to `~/.config/transcribe_app/settings.json`

## Requirements

- Python 3.12+
- A CUDA-capable GPU is recommended for real-time performance; CPU-only mode is supported
- PortAudio (required by `sounddevice`; usually pre-installed on Linux)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd whisperlivekit_gui

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### PortAudio on non-Linux platforms

| Platform | Install command |
|---|---|
| macOS | `brew install portaudio` |
| Windows | bundled with most `sounddevice` wheels — no extra step needed |

## Running

```bash
source .venv/bin/activate
python -m transcribe_app
```

A legacy single-file prototype is also preserved:

```bash
python transcribe_app.py
```

## Engine Backends

| Engine | Latency | Best for |
|---|---|---|
| **WhisperLive (streaming)** | Very low — incremental updates while you speak | Dictation, live captions |
| **Faster Whisper (VAD)** | Low — transcribes after each speech segment | Cleaner output, lower GPU load |

Select the backend in **Edit → Settings → Engine**.

## Model Selection

Models are chosen automatically based on language, speed preference, and GPU availability:

| Language | Fast (CPU) | Normal (CPU) | Fast (GPU) | Normal (GPU) |
|---|---|---|---|---|
| English | `small.en` | `medium.en` | `medium.en` | `large-v3-turbo` |
| Deutsch | `small` | `medium` | `medium` | `large-v3-turbo` |

Models are downloaded on first use and cached under `~/.cache/huggingface/hub/`.

## Voice Commands

Speak these words to insert formatting without touching the keyboard:

| Spoken (English) | Spoken (Deutsch) | Output |
|---|---|---|
| "newline" | "neue Zeile" | line break |
| "new paragraph" | "neuer Absatz" | blank line |
| "heading …" | "Überschrift …" | `# …` (at sentence start) |
| "period" | "Punkt" | `. ` |
| "comma" | "Komma" | `, ` |
| "question mark" | "Fragezeichen" | `? ` |
| "exclamation mark" | "Ausrufezeichen" | `! ` |
| "colon" | "Doppelpunkt" | `: ` |
| "semicolon" | "Semikolon" | `; ` |
| "hyphen" / "dash" | "Bindestrich" | `-` |
| "firstly" … "tenth" | "erstens" … "zehntens" | `1.` … `10.` |

## Settings

Open **Edit → Settings** to configure:

- **Language** — English or Deutsch
- **Model** — Fast or Normal speed
- **Engine** — WhisperLive or Faster Whisper
- **Device** — CUDA or CPU
- **Interface** — English or German UI
- **Style Prompt** — custom text prepended to each Whisper inference
- **ASR post-processing** — re-transcribe the session recording on stop

Settings are saved to:

- **Linux**: `~/.config/transcribe_app/settings.json`
- **Windows**: `%APPDATA%\transcribe_app\settings.json`

## Post-processing / Stop sequence

Clicking **Stop** (or releasing Space in push-to-talk mode) does not immediately end processing. The sequence is:

1. The microphone stream is closed and a flush signal is sent to the Whisper processor.
2. The Whisper model continues running until all queued audio has been transcribed.
3. Once the model signals completion, the UI receives the `finalise` event.
4. If ASR post-processing is enabled, the full session audio is re-transcribed and a diff is saved.

The Record button remains disabled until this sequence is complete.

## Project Layout

```
transcribe_app/
├── __main__.py                   # entry point
├── config.py                     # audio constants, language/model definitions
├── settings.py                   # Settings dataclass, JSON load/save
├── text_processing.py            # voice command substitution, text cleaning
├── i18n.py                       # UI localisation (en / de)
├── engine_protocol.py            # EngineManagerProtocol ABC + factory
├── engine_manager.py             # WhisperLiveKit streaming backend
├── engine.py                     # asyncio loop, model lifecycle, audio stream
├── alternative_engine_manager.py # Faster Whisper VAD backend
├── alternative_engine.py         # VAD + faster-whisper inference
├── session_file_manager.py       # session audio recording
└── ui/
    ├── theme.py                  # colour tokens, widget helpers, ttk style
    ├── main_window.py            # TranscriptionApp — view + event wiring
    ├── mic_test.py               # MicTestWindow — live level meter
    └── dialogs/
        └── settings_dialog.py    # Settings dialog
```

## Building a standalone executable

- **Thread model**: Tkinter runs on the main thread. A daemon thread hosts an `asyncio` event loop that owns the engine and model. Audio is captured by a `sounddevice` callback and fed to the engine via `asyncio.run_coroutine_threadsafe`. Results are returned to Tkinter through a `queue.Queue` polled every 50 ms via `root.after`.
- **Lazy imports**: `sounddevice`, `AudioProcessor`, and engine classes are imported inside the background thread to avoid PortAudio / CUDA hangs at window open.
- **Immutable settings**: `Settings` is a dataclass; saving produces a new instance via `dataclasses.replace()` rather than mutating state.
- **Engine abstraction**: Both backends implement `EngineManagerProtocol`; the UI has no direct dependency on either concrete class.

## Building a Standalone Executable

> **Note:** PyInstaller cannot cross-compile. Build on the target OS.
> The resulting executable is ~1–2 GB because PyTorch is bundled.
> Whisper models are **not** bundled — they are downloaded to
> `~/.cache/huggingface/hub/` on first launch.

### Prerequisites (both platforms)

```bash
pip install pyinstaller
```

### Linux

```bash
pyinstaller transcribe_app_linux.spec
```

The binary is written to `dist/transcribe_app`. Make it executable and run:

```bash
chmod +x dist/transcribe_app
./dist/transcribe_app
```

**Runtime dependencies on the target Linux machine:**


| Dependency                            | Install                                          |
| ------------------------------------- | ------------------------------------------------ |
| PortAudio (for sounddevice)           | `sudo apt install libportaudio2`                 |
| Java 8+ (for LanguageTool, optional)  | `sudo apt install default-jre`                   |
| Dependency | Install |
|---|---|
| PortAudio | `sudo apt install libportaudio2` |
| Java 8+ (for LanguageTool, optional) | `sudo apt install default-jre` |

### Windows

Run the following in a Command Prompt or PowerShell **on a Windows machine**
(or in a Windows GitHub Actions runner — see CI section below):

```bat
pyinstaller transcribe_app_windows.spec
```

The binary is written to `dist\transcribe_app.exe` — no installer required.

For the faster-starting onedir build:

```bat
pyinstaller transcribe_app_windows_onedir.spec
```

The app is written to `dist\transcribe_app\transcribe_app.exe`. When you
zip it, upload it, or include it in an installer, include the whole
`dist\transcribe_app` directory recursively. Do not install only
`transcribe_app.exe`; the executable needs the sibling `_internal` directory,
including `_internal\_tcl_data` and `_internal\_tk_data`, for Tkinter to start.

If Windows shows an error like:

```text
Tcl data directory "...\\transcribe_app\\_internal\\_tcl_data" not found
```

the installed folder is incomplete. Rebuild with the onedir spec and reinstall
the full `dist\transcribe_app` folder and allow Permissions "Vollzugriff" on windows.

**Runtime dependencies on the target machine:**

| Dependency | Notes |
|---|---|
| Visual C++ Redistributable | Usually pre-installed; download from Microsoft if missing |
| Java 8+ (for LanguageTool, optional) | Add `java.exe` to `PATH` |

### Troubleshooting the build

- **Missing module at runtime**: set `console=True` in the spec file to see the traceback, then add the module to `hiddenimports`.
- **UPX errors on Windows**: set `upx=False` in the spec file if UPX corrupts a torch CUDA DLL.
- **App opens then immediately closes**: set `console=True` temporarily to read the startup error.

### Building a Windows executable via GitHub Actions

```yaml
name: Build Windows executable
on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt pyinstaller
      - name: Build
        run: pyinstaller transcribe_app_windows.spec
      - uses: actions/upload-artifact@v4
        with:
          name: transcribe_app-windows
          path: dist/transcribe_app.exe
```

For the onedir build, upload the directory instead:

```yaml
      - name: Build onedir
        run: pyinstaller transcribe_app_windows_onedir.spec
      - uses: actions/upload-artifact@v4
        with:
          name: transcribe_app-windows-onedir
          path: dist/transcribe_app/**
```

## Development

```bash
# Run tests
pytest
```
