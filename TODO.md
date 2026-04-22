# TODO

Code review findings, ordered by severity.

---

## Bugs

### 1. Re-transcription blocks the UI thread

`_postprocess()` is called from `_poll_ui()` on the Tk main thread. Inside it,
`_do_retranscribe()` creates a fresh `SpeechToTextEngine` (loading the model),
reads all WAV files, and runs full inference — potentially blocking for 10–60 s.

`_RETRANSCRIBE_TIMEOUT_S = 120`, the `_retranscribing` flag, and the
imported-but-unused `concurrent.futures` are remnants of a previous design that
ran this in a thread. `_on_retranscribe_done()` (`main_window.py:660–693`) is the
old UI-thread callback — it is never called and is dead code.

**Fix:** move `_do_retranscribe` into a `ThreadPoolExecutor`, schedule the result
back via `root.after(0, ...)`, delete `_on_retranscribe_done`, and remove the
`concurrent.futures` import.

---

### 2. `main_window.py` bypasses the engine protocol abstraction

The docstring says the window delegates engine concerns to `EngineManagerProtocol`,
but lines 33 and 629–631 directly import and instantiate the concrete
`SpeechToTextEngine` and `Config` from `alternative_engine`:

```python
from ..alternative_engine import Config, SpeechToTextEngine, LANGUAGE_TRANSLATION
...
engine = SpeechToTextEngine(asr_cfg)
retranscribed = engine.transcribe_internal(audio)
```

This hard-wires re-transcription to the VAD backend regardless of which engine is
active. `EngineManagerProtocol.whisper_asr` exists precisely for post-session
re-transcription — re-transcription should call through that property, not
construct its own engine. Currently the feature silently does nothing when
WhisperLive is selected.

---

### 3. Audio capture is silently broken for the `faster_whisper` backend

`SessionFileManager` is attached to `engine.audio_sink` for WAV capture, but
`AlternativeEngineManager` explicitly documents that `audio_sink` is
"API compatibility only" and not wired to the engine's audio path (the engine owns
its own stream). WAV files are never written for `faster_whisper`, so
`_do_retranscribe` finds an empty `session_mgr.wav_paths` and silently does
nothing. Either wire the capture correctly for that backend or disable
re-transcription for it with a user-visible message.

---

### 4. `_do_retranscribe` reconstructs paths it already has, via a private attribute

```python
# main_window.py:638
sr, chunk = load_wav_as_array(wav_filename=os.path.join(session_mgr._wav_dir, wav_path.name))
```

`wav_path` is already an absolute `Path` (set in `SessionFileManager._open_wav()`).
The line accesses `session_mgr._wav_dir` (private) to reconstruct the same path.
Replace with:

```python
sr, chunk = wavfile.read(wav_path)
```

While here: `load_wav_as_array` is a one-liner wrapper for `wavfile.read` sitting
in `main_window.py` with no business there — inline it or move it to
`session_file_manager.py`.

---

### 5. `logging.basicConfig` runs at import time in `alternative_engine.py`

Lines 65–69 call `logging.basicConfig(...)` at module scope. This fires the moment
the module is imported (inside the background thread, after the window opens) and
overrides any logging configuration set up by the application. Move the call into
the `if __name__ == "__main__"` block at the bottom, which is where it belongs for
a standalone-runnable module.

---

## Design issues

### 6. `Settings` is mutable despite being documented as immutable

`@dataclass` without `frozen=True` means the "replaced, not mutated" contract is
unenforced. Add `@dataclass(frozen=True)` to `settings.py` so the type checker and
runtime catch accidental mutations.

---

### 7. `apply_commands_full` returns `None` for "no change" — wrong API

The `-> str | None` return forces every caller to write boilerplate:

```python
processed = apply_commands_full(clean(new_text))
new_text_final = processed if processed is not None else new_text
if processed is not None:
    # redraw
```

The `None` sentinel encodes a UI concern (skip redraw) inside a pure text function.
Change the return type to `str` (always return the result string). Callers that need
to know whether anything changed can compare `result != original`.

---

### 8. `strip_prompt_leak` compiles a regex on every call

```python
re.sub(re.escape(prompt), " ", text, flags=re.IGNORECASE)
```

Called twice per `_set_text()` invocation (every 50 ms during recording). The
prompt never changes during a session. Cache the compiled pattern, e.g. with
`@functools.lru_cache` keyed on the escaped prompt string.

---

### 9. `_begin_write` / `_end_write` should be a context manager

The enable/disable pair is called in 8 places. An exception between them leaves
the widget permanently in the wrong state. Replace with a context manager:

```python
@contextlib.contextmanager
def _write_ctx(self):
    self._text.config(state=tk.NORMAL)
    try:
        yield
    finally:
        if self._recording:
            self._text.config(state=tk.DISABLED)
```

---

### 10. Menu updates use fragile positional indices

```python
self._menubar.entryconfigure(0, label=t("menu.file"))
self._menubar.entryconfigure(1, label=t("menu.edit"))
self._file_menu.entryconfigure(0, label=t("menu.file.save"))
```

These break silently if menu items are reordered or new items are inserted. Store
each menu item reference or use a small dict keyed by i18n key so `_apply_ui_lang`
updates by name rather than by position.

---

### 11. `vad_silence_gap` is not declared in `EngineManagerProtocol`

`main_window.py` sets `self._mgr.vad_silence_gap = ...` unconditionally, but the
attribute is absent from the protocol. For `EngineManager` (WhisperLive) this
silently creates an orphaned instance attribute. Either add the attribute to the
protocol with a no-op default, or gate the assignment with `isinstance`.

---

### 12. `LANGUAGE_TRANSLATION` in `alternative_engine.py` is redundant

```python
LANGUAGE_TRANSLATION = {"Deutsch": "de", "English": "en"}
```

This mapping is imported into `main_window.py`. The same data already exists as
`LANGUAGE_OPTS[lang]["lan"]` in `config.py`. Replace the usage in
`main_window.py:625` with `LANGUAGE_OPTS[self._settings.language]["lan"]` and
delete `LANGUAGE_TRANSLATION`.

---

## Minor issues

### 13. Ordinal voice commands miss start-of-text position

All ordinal patterns require leading whitespace (`\s+\b`):

```python
re.compile(r"\s+\berstens\b[,.]?\s*", re.I)
```

`apply_commands_full("erstens")` — the ordinal spoken as the very first word —
returns `None` with no match. Use `(?:^|\s+)` as the heading patterns already do.

---

### 14. Platform-specific path constants are duplicated across modules

`_SESSIONS_DIR` and `_TRANSCRIPTION_DIFF_DIR` in `main_window.py` and
`_SNIPPET_DIR` in `alternative_engine_manager.py` are all derived from
`%APPDATA%/transcribe_app/…` with the same platform switch. Define all app
data paths once in `config.py` (or a dedicated `paths.py`) so a platform change
is a single edit.
