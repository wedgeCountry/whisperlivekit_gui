import multiprocessing
import logging
import os
import sys
import tkinter as tk


def _get_work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    """Return the usable desktop work area as (left, top, width, height)."""
    if sys.platform == "win32":
        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            rect = RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        except Exception:
            pass

    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def _apply_initial_window_geometry(root: tk.Tk) -> None:
    """Open the main window centered in the visible work area with a broad aspect ratio."""
    root.update_idletasks()
    left, top, work_w, work_h = _get_work_area(root)
    min_w = 760
    min_h = 520
    ratio = 1.2
    target_h = max(min_h, int(work_h * 0.78))
    target_w = max(min_w, int(target_h * ratio))
    width = min(work_w - 40, target_w)
    height = min(work_h - 40, max(min_h, int(width / ratio)))
    x = left + max(0, (work_w - width) // 2)
    y = top + max(0, (work_h - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")


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
    from transcribe_app.recording_cleanup import delete_all_recording_artifacts
    from transcribe_app.ui.main_window import TranscriptionApp
    logging.getLogger("whisperlivekit").setLevel(logging.WARNING)
    logging.getLogger("whisperlivekit.audio_processor").setLevel(logging.WARNING)
    delete_all_recording_artifacts()
    root = tk.Tk()
    _apply_initial_window_geometry(root)
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _fix_frozen_streams()
    main()
