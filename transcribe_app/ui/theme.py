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
C_INPUT     = "#fbfdff"

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
    style.configure(
        "Modern.TCombobox",
        fieldbackground=C_SURFACE,
        background=C_SURFACE,
        foreground=C_TEXT,
        bordercolor=C_BORDER,
        lightcolor=C_BORDER,
        darkcolor=C_BORDER,
        arrowcolor=C_TEXT,
        arrowsize=16,
        padding=(10, 6),
        relief="flat",
    )
    style.map(
        "Modern.TCombobox",
        fieldbackground=[("readonly", C_SURFACE)],
        foreground=[("readonly", C_TEXT)],
        bordercolor=[("focus", C_ACCENT), ("readonly", C_BORDER), ("disabled", C_BORDER)],
        lightcolor=[("focus", C_ACCENT)],
        darkcolor=[("focus", C_ACCENT)],
        arrowcolor=[("disabled", C_MUTED)],
    )
    style.configure(
        "Modern.TSpinbox",
        fieldbackground=C_SURFACE,
        background=C_SURFACE,
        foreground=C_TEXT,
        bordercolor=C_BORDER,
        lightcolor=C_BORDER,
        darkcolor=C_BORDER,
        arrowsize=14,
        padding=(8, 5),
        relief="flat",
    )
    style.map(
        "Modern.TSpinbox",
        bordercolor=[("focus", C_ACCENT)],
        lightcolor=[("focus", C_ACCENT)],
        darkcolor=[("focus", C_ACCENT)],
    )
    style.configure(
        "Modern.Vertical.TScrollbar",
        background="#cbd5e1",
        troughcolor=C_SURFACE,
        bordercolor=C_SURFACE,
        arrowcolor="#64748b",
        darkcolor="#cbd5e1",
        lightcolor="#cbd5e1",
        relief="flat",
        gripcount=0,
        arrowsize=12,
        width=10,
    )
    style.map(
        "Modern.Vertical.TScrollbar",
        background=[("active", "#94a3b8"), ("pressed", "#64748b")],
        arrowcolor=[("active", "#475569")],
    )


def make_card(
    parent: tk.Widget,
    *,
    bg: str = C_SURFACE,
    border: str = C_BORDER,
    padx: int = 0,
    pady: int = 0,
) -> tk.Frame:
    return tk.Frame(
        parent,
        bg=bg,
        highlightbackground=border,
        highlightthickness=1,
        bd=0,
        padx=padx,
        pady=pady,
    )


def style_text_widget(widget: tk.Text) -> None:
    widget.configure(
        bg=C_INPUT,
        fg=C_TEXT,
        insertbackground=C_TEXT,
        selectbackground=C_ACCENT,
        selectforeground="#ffffff",
        relief=tk.FLAT,
        bd=0,
        highlightthickness=0,
    )


def style_scale_widget(widget: tk.Scale, *, trough: str = C_STATUS_BG) -> None:
    widget.configure(
        bg=C_SURFACE,
        fg=C_TEXT,
        troughcolor=trough,
        activebackground=C_ACCENT,
        sliderrelief=tk.FLAT,
        sliderlength=18,
        highlightbackground=C_SURFACE,
        highlightcolor=C_SURFACE,
        highlightthickness=0,
    )
    # Tk uses the widget background for the slider/thumb on some platforms, so
    # give it a slightly darker neutral to keep it visible against white cards.
    widget.configure(
        bg="#cbd5e1",
        activebackground="#94a3b8",
        highlightthickness=0,
        relief=tk.FLAT,
        bd=0,
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
