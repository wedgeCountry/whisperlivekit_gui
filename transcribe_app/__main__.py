import logging
import tkinter as tk

def main() -> None:
    from transcribe_app.ui.main_window import TranscriptionApp
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("whisperlivekit").setLevel(logging.WARNING)
    logging.getLogger("whisperlivekit.audio_processor").setLevel(logging.WARNING)
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
