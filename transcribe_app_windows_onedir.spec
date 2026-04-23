# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Windows (x86_64), onedir layout.
# Build on Windows with:
#   pip install -r requirements-dev.txt
#   pyinstaller transcribe_app_windows_onedir.spec
#
# This keeps DLLs/data next to the executable instead of packing everything
# into one self-extracting file. It usually starts much faster than onefile.

from PyInstaller.utils.hooks import collect_all

# Collect all files (Python, data, DLLs) from heavy packages.
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
        # Lazy imports in engine.py.
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
        # Lazy import in engine.py open_mic_stream().
        'sounddevice',
        '_sounddevice',
        # Lazy import in engine.py shutdown().
        'torch',
        'torch.cuda',
        # Correction backends loaded on demand.
        'language_tool_python',
        'spylls',
        'spylls.hunspell',
        'spylls.hunspell.data',
        'httpx',
        # Tkinter submodules used in ui/.
        'tkinter',
        'tkinter.ttk',
        'tkinter.font',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.scrolledtext',
        # SciPy / NumPy used by audio and optional retranscription paths.
        'scipy',
        'scipy.signal',
        'scipy.io.wavfile',
        'numpy',
        # Windows-specific: needed for sounddevice / PortAudio.
        'cffi',
        '_cffi_backend',
    ] + hiddenimports_wlk + hiddenimports_fw + hiddenimports_ct2
      + hiddenimports_ort + hiddenimports_app,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torchvision',
        'torchaudio',
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
    [],
    exclude_binaries=True,
    name='transcribe_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # Do not compress torch DLLs; UPX can corrupt them.
        'torch_cpu.dll',
        'torch_cuda.dll',
        'cublas64_*.dll',
        'cudnn64_*.dll',
    ],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'torch_cpu.dll',
        'torch_cuda.dll',
        'cublas64_*.dll',
        'cudnn64_*.dll',
    ],
    name='transcribe_app',
)
