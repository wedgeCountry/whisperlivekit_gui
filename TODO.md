# TODO

Prioritised by severity — highest priority first.

---

## #1 Fix 10 failing tests in `test_text_processing.py`

Regex ordinal patterns (`firstly`…`tenth`, `erstens`…`zehntens`) require a leading `\s+`
anchor and therefore never match at the start of a sentence. Several punctuation
substitutions produce double spaces. `strip_prompt_leak()` collapses the spaces
surrounding a removed prompt match. These are silent production bugs, not just
broken tests. Fix the bugs, leafe the tests as is

---
---

## #3 Move grammar correction off the UI thread

`_launch_grammar()` in `main_window.py` calls the blocking `LanguageTool.correct()`
directly on the Tkinter main thread, hard-freezing the window until the call returns.
Run correction in a background thread with a timeout (≈10 s) and schedule
`_on_grammar_done()` back to the UI thread via `root.after()`.

---

## #7 Add file-based logging (`debug.log`)

No log file exists anywhere in the app. All diagnostics go through `on_status()`,
which only ever shows the last message. Add a rotating `debug.log` (max 2 MB) under
the config directory and route all currently swallowed exceptions to at least
`logging.warning` / `logging.error`. This is essential for triaging user-reported bugs. 

---

## #8 Reduce default socket timeout from 600 s to 60 s

`engine.py` sets `socket.setdefaulttimeout(600)`, meaning a hung model-download
server blocks the entire app for up to 10 minutes before any error surfaces. Lower to
60 s so failures are reported promptly.

---

## #9 Remove unused `diarization` import in `engine.py`

`engine.py` line 21 imports `from whisperlivekit import diarization` but the symbol
is never used. The dead import causes unnecessary import-time work and may trigger
side effects inside whisperlivekit.

---

## #10 Centralise the language → grammar-code mapping

`GRAMMAR_LANG_CODES` is referenced in `config.py`, `grammar.py`, and inside a local
import within `settings_dialog._update_grammar_state()`. Keep the canonical mapping
in `config.py` and import it from there everywhere. Promote the local import to
module level in `settings_dialog.py`.
