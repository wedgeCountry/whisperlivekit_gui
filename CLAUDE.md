# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A single-file Tkinter GUI application (`transcribe_app.py`) for live speech transcription powered by [WhisperLiveKit](https://github.com/QUENTINFUXA/WHISPERLIVEKIT). Supports English and German, with optional post-processing via Ollama, LanguageTool, or Spylls.

## Running the app

```bash
source .venv/bin/activate
python transcribe_app.py
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

## Architecture

The entire application lives in `transcribe_app.py`. Key design:

- **Thread model**: Main thread runs Tkinter UI. A dedicated background thread hosts an `asyncio` event loop that owns `TranscriptionEngine` and `AudioProcessor` (from `whisperlivekit`). A `sounddevice` input stream callback feeds raw PCM via `asyncio.run_coroutine_threadsafe`. A `threading.Queue` bridges async transcription results back to the Tkinter main thread, polled every 50 ms.
- **`TranscriptionApp`**: The sole class. Owns all state — recording state, the async loop/engine/processor references, accumulated transcript text, and settings.
- **Lazy imports**: `sounddevice`, `AudioProcessor`, `TranscriptionEngine`, and `WhisperLiveKitConfig` are imported lazily (at mic-stream start) to avoid a PortAudio hang on startup.
- **Display model**: Committed (finalised) transcript lines accumulate in the text area with normal style. The in-progress live buffer is shown in muted italics (`"buffer"` tag) and replaced on each update.
- **Post-processing pipeline**: After WhisperLiveKit commits a segment, it optionally passes through `_apply_commands` (voice commands → markdown), then one of three correction backends: Ollama (async HTTP to local LLM), LanguageTool (via `language_tool_python`), or Spylls (offline Hunspell).
- **Settings persistence**: Saved as JSON to `~/.config/transcribe_app/settings.json` (Linux) or `%APPDATA%\transcribe_app\settings.json` (Windows). Stores language, prompts, system prompts, Ollama models, and correction backend.
- **GPU detection**: `torch.cuda.is_available()` at import time; selects `large-v3-turbo` (GPU) vs `medium`/`medium.en` (CPU) model.

## Voice commands (spoken → markdown)

| Spoken | Output |
|---|---|
| "newline" / "neue Zeile" | `\n` |
| "new paragraph" / "neuer Absatz" | `\n\n` |
| "heading …" / "Überschrift …" (at sentence start) | `\n# …` |

## Model cache location

WhisperLiveKit models are cached under `~/.cache/huggingface/hub/` in `models--Systran--faster-whisper-*` or `models--mobiuslabsgmbh--faster-whisper-*` directories.
