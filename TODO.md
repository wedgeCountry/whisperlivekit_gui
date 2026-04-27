# TODO

## Race conditions / correctness

- [x] **Language/speed change bypasses drain guard** — `_on_language_change()` and `_on_speed_change()` call `_reload_engine()` without checking `_session_draining`. If the user changes language or speed immediately after clicking Stop while the session is still draining, a reload starts while the old session is still running. The settings dialog defers via `_pending_engine_action` but these two paths do not. (`ui/main_window.py:363–378`)

- [x] **`AlternativeEngineManager.shutdown()` races `_poll_results`** — `shutdown()` sets `self._engine = None` without joining `_poll_results` first. If the thread is between its `if self._engine is not None:` check and the `flush_done.wait()` call when `shutdown()` runs, `flush_done.wait()` is skipped, the queue is drained early, and `_on_finalise` fires with partial results. (`alternative_engine_manager.py:229–232`)

- [x] **`SessionFileManager._open_wav()` appends path before file creation** — the path is added to `_wav_paths` before `wave.open()` is called. If `wave.open()` raises, a non-existent path remains in `wav_paths`; `_do_retranscribe` logs a warning and skips it, but the caller never knows the file was never created. Fix: append only after the file is successfully opened. (`session_file_manager.py:77–84`)

- [x] **`EngineManager.stop_session()` final flush may be dropped on fast close** — `process_audio(None)` is sent via `run_coroutine_threadsafe` and the call returns immediately. If the user closes the window before the coroutine runs, `shutdown()` calls `cancel_pending_tasks()` and stops the loop first, silently dropping the final audio flush. (`engine_manager.py:196–200`)

## Indefinite hangs with no recovery path

- [x] **No timeout on WhisperLive `results_gen` loop** — `AsyncEngine._run_session()` iterates `async for front_data in results_gen:` with no timeout. If WhisperLiveKit's `AudioProcessor` hangs after receiving `process_audio(None)`, the generator never closes, `_on_finalise` never fires, and `_session_draining` stays `True` forever. The user cannot record again without restarting the app. (`engine.py:378`)

- [x] **Executor timeout leaves zombie thread** — `asyncio.wait_for(run_in_executor(...), timeout=...)` cancels the asyncio task but cannot stop the underlying thread. If model building or warmup deadlocks in a frozen exe, the thread runs indefinitely. The app shows an error and recovers, but the thread is unrecoverable until process exit. (`engine.py:247, 303`)

## UI state bugs

- [ ] **Deferred status-bar timers overwrite later messages** — `_copy_text()`, `_save_file()`, and `_load_file()` schedule `root.after()` callbacks that restore the status bar, but never save or cancel the callback ID. If re-transcription or an error is displayed within the timer window, the callback silently overwrites it with "Ready" or "Recording". (`ui/main_window.py:793, 812, 834`)

- [ ] **`_on_close()` does not cancel pending `root.after()` callbacks** — the timers above (and any others) fire after `root.destroy()` and raise `TclError` on destroyed-widget access. Fix: store the after-IDs and cancel them in `_on_close()`. (`ui/main_window.py:926–928`)

## Logging / observability

- [ ] **`_TqdmCapture.isatty()` returns `True`, polluting the log file** — tqdm writes `\r`-prefixed progress lines. In the frozen-exe log file these appear as raw carriage-return characters, making the download-progress section unreadable. Override `isatty()` to return `False`, or strip `\r` in `write()`. (`engine.py:60`)

- [ ] **Root logger set to `DEBUG` in frozen exe** — `_fix_frozen_streams()` sets the root logger level to `DEBUG`, making ctranslate2, faster-whisper, and sounddevice very verbose. The log file grows large quickly. Set root to `INFO` and apply `DEBUG` only to `transcribe_app.*` loggers. (`__main__.py:_fix_frozen_streams`)

## Packaging / spec

- [ ] **`TOKENIZERS_PARALLELISM` not set** — HuggingFace tokenizers spawn parallel threads by default. In a frozen exe alongside ctranslate2, this can trigger the same OpenMP double-init deadlock that `OMP_NUM_THREADS` is meant to prevent. Add `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` to `_pin_threads()`. (`engine_manager.py:38–44`)

- [ ] **UPX exclude patterns for ctranslate2 DLLs are guesses** — the actual DLL names bundled by `ctranslate2` depend on the installed version. If they differ from the patterns in the spec, those DLLs still get UPX-compressed and can deadlock. After the first build, inspect `dist/transcribe_app/_internal/` and replace the pattern list with exact filenames. (`transcribe_app_windows_onedir.spec:91–104`)

## Code quality / minor

- [ ] **`_postprocess()` docstring is outdated** — says *"the record button is re-enabled straight away with the live text visible"*, which was true in an older version. `_session_draining` now stays `True` during re-transcription, so the button is only re-enabled when re-transcription completes. (`ui/main_window.py:607`)

- [ ] **Worker thread in `run_with_stream` is not joined** — `Thread(target=self.worker, daemon=True).start()` keeps no reference. The worker cannot be force-stopped or joined during shutdown. For a long-running process with many sessions, each session's worker thread must run to natural completion with no external control. (`alternative_engine.py:354`)

- [ ] **Inconsistent flush timeouts** — `run_with_stream` waits 10 s for `flush_done` before stopping the event loop, but `_poll_results` waits up to 60 s for the same event. For a final speech segment whose transcription takes more than 10 s, the loop stops before the worker finishes; `_poll_results` still recovers via its own timeout, but the two values should be documented together to make the intent clear. (`alternative_engine.py:362–374`)

- [ ] **10 pre-existing test failures in `test_text_processing.py`** — `clean()`, `apply_commands_full()`, and `strip_prompt_leak()` are exercised by these tests and used in the live display path. The failures hide any regressions introduced in that module and should be resolved.
