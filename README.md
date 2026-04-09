# whisperlivekit_gui
A tkinter GUI for whisperlivekit

https://github.com/QUENTINFUXA/WHISPERLIVEKIT

Implementierung mit Hilfe von Claude Code (Sonnet 4.5, Pro-Plan)

---

## Running from source

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m transcribe_app
```

**Linux** — optional Spylls spell-checking requires system Hunspell dictionaries:

```bash
sudo apt install hunspell-en-us hunspell-de-de
```

**Ollama** (optional LLM post-processing): install and start Ollama, then select an
Ollama model in the app settings. The app connects to `http://localhost:11434`.

**LanguageTool** (optional grammar correction): requires Java 8+ on `PATH`. The
`language_tool_python` library downloads the LanguageTool server JAR on first use
(~200 MB, cached in `~/.cache/language_tool_python/`).

---

## Building a standalone executable

> **Important:** PyInstaller cannot cross-compile. Build on the target OS.
> The resulting executable is ~1–2 GB because PyTorch is bundled.
> Whisper models are **not** bundled — they are downloaded to
> `~/.cache/huggingface/hub/` on first launch.

### Prerequisites (both platforms)

```bash
pip install pyinstaller
```

### Linux

```bash
pyinstaller transcribe_app_linux.spec
```

The binary is written to `dist/transcribe_app`. Make it executable and run:

```bash
chmod +x dist/transcribe_app
./dist/transcribe_app
```

**Runtime dependencies on the target Linux machine:**

| Dependency | Install |
|---|---|
| PortAudio (for sounddevice) | `sudo apt install libportaudio2` |
| Java 8+ (for LanguageTool, optional) | `sudo apt install default-jre` |
| Hunspell dicts (for Spylls, optional) | `sudo apt install hunspell-en-us hunspell-de-de` |

### Windows

Run the following in a Command Prompt or PowerShell **on a Windows machine**
(or in a Windows GitHub Actions runner — see CI section below):

```bat
pyinstaller transcribe_app_windows.spec
```

The binary is written to `dist\transcribe_app.exe`. No installer is required —
copy the single `.exe` to any Windows 10/11 machine and run it.

**Runtime dependencies on the target Windows machine:**

| Dependency | Notes |
|---|---|
| Visual C++ Redistributable | Usually pre-installed; download from Microsoft if missing |
| Java 8+ (for LanguageTool, optional) | Add `java.exe` to `PATH` |

### Troubleshooting the build

- **Missing module at runtime**: run the build with `console=True` in the spec
  file to see the traceback, then add the missing module to `hiddenimports`.
- **`ModuleNotFoundError: No module named 'X'`**: add `'X'` to the
  `hiddenimports` list in the relevant `.spec` file and rebuild.
- **UPX errors on Windows**: set `upx=False` in the spec file if UPX corrupts
  a DLL (most often a torch CUDA DLL).
- **App opens then immediately closes**: set `console=True` temporarily to read
  the startup error.

### Building Windows executable via GitHub Actions

Create `.github/workflows/build-windows.yml`:

```yaml
name: Build Windows executable
on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt pyinstaller
      - name: Build
        run: pyinstaller transcribe_app_windows.spec
      - uses: actions/upload-artifact@v4
        with:
          name: transcribe_app-windows
          path: dist/transcribe_app.exe
```