"""Design tokens and reusable widget factories. No business logic."""

import tkinter as tk
from tkinter import ttk

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG        = "#f8fafc"
C_SURFACE   = "#ffffff"
C_HEADER    = "#ffffff"
C_BORDER    = "#e2e8f0"
C_TEXT      = "#1e293b"
C_MUTED     = "#64748b"
C_ACCENT    = "#3b82f6"
C_ACCENT_H  = "#2563eb"
C_DANGER    = "#ef4444"
C_DANGER_H  = "#dc2626"
C_BTN       = "#ffffff"
C_BTN_H     = "#f1f5f9"
C_STATUS_BG = "#f1f5f9"
C_BUFFER    = "#94a3b8"

# ── Typography ─────────────────────────────────────────────────────────────────
F_UI    = ("TkDefaultFont", 10)
F_MONO  = ("TkFixedFont",   11)
F_SMALL = ("TkDefaultFont",  9)
F_LABEL = ("TkDefaultFont", 10)


def center_on_parent(win: tk.Toplevel, parent: tk.Widget) -> None:
    """Position *win* centered over *parent* after its geometry is known."""
    win.update_idletasks()
    pw = parent.winfo_rootx() + parent.winfo_width()  // 2
    ph = parent.winfo_rooty() + parent.winfo_height() // 2
    w  = win.winfo_width()
    h  = win.winfo_height()
    win.geometry(f"+{pw - w // 2}+{ph - h // 2}")


def apply_ttk_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground=C_SURFACE,
        background=C_SURFACE,
        foreground=C_TEXT,
        bordercolor=C_BORDER,
        lightcolor=C_BORDER,
        darkcolor=C_BORDER,
        arrowcolor=C_MUTED,
        arrowsize=14,
        padding=(6, 4),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", C_SURFACE)],
        foreground=[("readonly", C_TEXT)],
        bordercolor=[("focus", C_ACCENT)],
    )


def hoverable(btn: tk.Button, normal: str, hover: str) -> None:
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda _e: btn.config(bg=normal))


def make_btn(
    parent: tk.Widget,
    text: str,
    command,
    *,
    primary: bool = False,
    danger: bool = False,
) -> tk.Button:
    if primary:
        bg, fg, hov = C_ACCENT, "#ffffff", C_ACCENT_H
    elif danger:
        bg, fg, hov = C_DANGER, "#ffffff", C_DANGER_H
    else:
        bg, fg, hov = C_BTN, C_TEXT, C_BTN_H

    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg,
        activebackground=hov, activeforeground=fg,
        font=F_UI, relief=tk.FLAT, bd=0,
        padx=20, pady=9, cursor="hand2",
        highlightbackground=C_BORDER,
        highlightthickness=1,
    )
    hoverable(btn, bg, hov)
    return btn
