import multiprocessing
import logging
import os
import sys
import tkinter as tk


def _fix_frozen_streams() -> None:
    """In a windowless (console=False) PyInstaller exe, sys.stdout and sys.stderr
    are None.  Any code that writes to them (tqdm, logging, print) would crash with
    AttributeError.  Redirect both to a log file next to the executable so that
    tqdm progress bars and error messages are preserved for debugging.
    """
    if not getattr(sys, "frozen", False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return

    try:
        log_dir = os.path.dirname(sys.executable)
        log_path = os.path.join(log_dir, "transcribe_app.log")
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    except OSError:
        # Fallback: discard all output rather than crash
        import io
        log_file = io.StringIO()  # type: ignore[assignment]

    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file


def main() -> None:
    from transcribe_app.ui.main_window import TranscriptionApp
    logging.getLogger("whisperlivekit").setLevel(logging.WARNING)
    logging.getLogger("whisperlivekit.audio_processor").setLevel(logging.WARNING)
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _fix_frozen_streams()
    main()
