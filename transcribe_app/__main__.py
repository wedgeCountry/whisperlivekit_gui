import tkinter as tk


def main() -> None:
    from transcribe_app.ui.main_window import TranscriptionApp
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
