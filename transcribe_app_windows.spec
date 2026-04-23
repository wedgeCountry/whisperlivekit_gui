# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Windows (x86_64)
# Build on Windows with:
#   pip install pyinstaller
#   pyinstaller transcribe_app_windows.spec

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all files (Python, data, DLLs) from heavy packages
datas_wlk, binaries_wlk, hiddenimports_wlk = collect_all('whisperlivekit')
datas_fw,  binaries_fw,  hiddenimports_fw  = collect_all('faster_whisper')
datas_ct2, binaries_ct2, hiddenimports_ct2 = collect_all('ctranslate2')
datas_ort, binaries_ort, hiddenimports_ort = collect_all('onnxruntime')
datas_app, binaries_app, hiddenimports_app = collect_all('transcribe_app')
icon_datas = [('assets/app_icon.png', 'assets')]

a = Analysis(
    ['transcribe_app/__main__.py'],
    pathex=['.'],
    binaries=binaries_wlk + binaries_fw + binaries_ct2 + binaries_ort + binaries_app,
    datas=datas_wlk + datas_fw + datas_ct2 + datas_ort + datas_app + icon_datas,
    hiddenimports=[
        # lazy imports in engine.py (imported inside background thread)
        'whisperlivekit',
        'whisperlivekit.config',
        'whisperlivekit.audio_processor',
        'whisperlivekit.whisper',
        'whisperlivekit.local_agreement',
        'whisperlivekit.silero_vad_iterator',
        'whisperlivekit.timed_objects',
        'whisperlivekit.diff_protocol',
        'whisperlivekit.tokens_alignment',
        'whisperlivekit.thread_safety',
        'faster_whisper',
        'ctranslate2',
        # lazy import in engine.py open_mic_stream()
        'sounddevice',
        '_sounddevice',
        # lazy import in engine.py shutdown()
        'torch',
        'torch.cuda',
        # correction backends (loaded on demand)
        'language_tool_python',
        'spylls',
        'spylls.hunspell',
        'spylls.hunspell.data',
        'httpx',
        # tkinter submodules used in ui/
        'tkinter',
        'tkinter.ttk',
        'tkinter.font',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.scrolledtext',
        # scipy / numpy (used by audio pipeline)
        'scipy',
        'scipy.signal',
        'scipy.io.wavfile',
        'numpy',
        # Windows-specific: needed for sounddevice / PortAudio
        'cffi',
        '_cffi_backend',
    ] + hiddenimports_wlk + hiddenimports_fw + hiddenimports_ct2
      + hiddenimports_ort + hiddenimports_app,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # safe to exclude: separate packages not installed in this venv
        'torchvision',
        'torchaudio',
        # not needed at runtime
        'matplotlib',
        'IPython',
        'jupyter',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='transcribe_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # do not compress torch DLLs — UPX can corrupt them
        'torch_cpu.dll',
        'torch_cuda.dll',
        'cublas64_*.dll',
        'cudnn64_*.dll',
    ],
    runtime_tmpdir=None,
    console=False,      # no console window; set True to see startup errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico',
)
