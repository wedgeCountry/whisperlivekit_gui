"""AsyncEngine — WhisperLiveKit model lifecycle and transcription sessions.

Designed to run entirely on a single asyncio event loop thread.  All thread
coordination (spawning, audio callbacks, UI bridging) lives in EngineManager
(engine_manager.py).  This module has no threading or sounddevice dependency.

Lazy globals
------------
_AudioProcessor, _TranscriptionEngine, _WhisperLiveKitConfig are None at import
time.  EngineManager._run_loop() populates them after the background thread
starts so PortAudio / CUDA never block the Tk window from opening.
"""

import asyncio
import logging
import sys
import threading
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)

from transcribe_app.config import GPU, IS_WINDOWS, LANGUAGE_OPTS
from transcribe_app.i18n import t
from transcribe_app.model_status import is_model_cached, loading_status

_AudioProcessor:       object = None  # type: ignore[assignment]
_TranscriptionEngine:  object = None  # type: ignore[assignment]
_WhisperLiveKitConfig: object = None  # type: ignore[assignment]


# ── Model cache helpers ────────────────────────────────────────────────────────

# ── tqdm progress capture ──────────────────────────────────────────────────────

class _TqdmCapture:
    """File-like object that intercepts tqdm progress lines and forwards them
    to a status callback.  Assigned to sys.stderr during model downloads so
    that the tqdm bar appears in the UI status bar instead of the terminal.
    """

    def __init__(self, on_status: Callable[[str], None], original: object) -> None:
        self._on_status = on_status
        self._original  = original

    def write(self, s: str) -> int:
        # tqdm writes lines like "\r82%|████| 376M/461M [00:13<00:12, 7.32MiB/s]"
        stripped = s.strip("\r\n ")
        if stripped and "%|" in stripped:
            self._on_status(stripped)
        elif stripped:
            self._original.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return True  # keep tqdm in \r-refresh mode so each write is a full bar update


# ── Inference helpers ──────────────────────────────────────────────────────────

def _make_warmup_audio() -> np.ndarray:
    """Return 1 s of Gaussian noise at speech-like RMS (≈ 0.07 normalised).

    Unlike silence (trivial encoder pass, 1–2 decoder steps), noise produces a
    dense mel spectrogram that exercises every attention head and drives the
    decoder through many token steps — the same code paths that run on real
    speech.  A fixed seed makes the warmup deterministic and reproducible.
    """
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 0.07).astype(np.float32)


def _has_avx512() -> bool:
    """Return True if CTranslate2 can use AVX-512 float16 on CPU.

    Without AVX-512, INT8 matrix multiplications fall back to AVX2/SSE4, which
    is noticeably slower on large models.  GPU users are unaffected.
    """
    try:
        import ctranslate2  # noqa: PLC0415
        return "float16" in ctranslate2.get_supported_compute_types("cpu")
    except Exception:
        _log.warning("Could not determine AVX-512 support", exc_info=True)
        return True  # assume OK; don't show a spurious warning


# ── AsyncEngine ────────────────────────────────────────────────────────────────

class AsyncEngine:
    """Model lifecycle and session management for the asyncio thread.

    EngineManager creates one instance per process lifetime, sets `loop` after
    the event loop is running, then calls `_load_engine` for the initial load
    and `schedule_reload` / `schedule_session` for subsequent operations.
    """

    def __init__(
        self,
        on_status:   Callable[[str], None],
        on_ready:    Callable[[bool], None],
        on_update:   Callable[[str, str], None],
        on_finalise: Callable[[str], None],
        on_open_mic: Callable[[], None],
        use_gpu:     bool,
    ) -> None:
        self._on_status   = on_status
        self._on_ready    = on_ready
        self._on_update   = on_update
        self._on_finalise = on_finalise
        self._on_open_mic = on_open_mic

        # Set by EngineManager after the event loop is created.
        self.loop: asyncio.AbstractEventLoop | None = None

        self.use_gpu:    bool          = use_gpu
        self._engine:    object | None = None
        self._processor: object | None = None
        self._lang:      str           = ""

        # Load-cancellation state — see EngineManager docstring for full explanation.
        self._exec_lock: threading.Lock      = threading.Lock()
        self._load_gen:  int                 = 0
        self._load_lock: asyncio.Lock | None = None
        self._load_task: asyncio.Task | None = None

        # Session-cancellation state.
        self._session_gen:  int                  = 0
        self._session_task: asyncio.Task | None  = None

    # ── Public interface for EngineManager ────────────────────────────────────

    @property
    def processor(self) -> "object | None":
        """The current AudioProcessor, or None between sessions."""
        return self._processor

    @property
    def whisper_asr(self) -> "object | None":
        """The FasterWhisperASR backend, or None if the model is not yet loaded.

        Exposes .transcribe(audio_float32, init_prompt="") and .original_language.
        Safe to read from the UI thread once a session has finished.
        """
        return getattr(self._engine, "asr", None)

    def schedule_reload(self, lang: str, prompt: str, speed: str) -> None:
        """Cancel any pending load and schedule a new one. Call from the event loop."""
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
        self._load_task = self.loop.create_task(  # type: ignore[union-attr]
            self._load_engine(lang, prompt, speed)
        )

    def schedule_session(self) -> None:
        """Cancel any draining session and schedule a new one. Call from the event loop."""
        if self._session_task is not None and not self._session_task.done():
            self._session_task.cancel()
        self._session_task = self.loop.create_task(self._run_session())  # type: ignore[union-attr]

    def cancel_pending_tasks(self) -> None:
        """Cancel all in-flight load and session tasks. Call from the event loop."""
        for task in (self._load_task, self._session_task):
            if task is not None and not task.done():
                task.cancel()

    def release(self) -> None:
        """Clear the TranscriptionEngine singleton and free GPU memory."""
        if self._engine is not None and _TranscriptionEngine is not None:
            _TranscriptionEngine._instance    = None  # type: ignore[union-attr]
            _TranscriptionEngine._initialized = False  # type: ignore[union-attr]
            self._engine = None
        if self.use_gpu:
            try:
                import torch  # noqa: PLC0415
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ── Model loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_wlk_config(size: str, lan: str, use_gpu: bool, prompt: str) -> object:
        """Build a WhisperLiveKitConfig for the given model size and language.

        Decoder and chunk-size choices are platform-specific:
        - Beam search on Windows CPU recovers from the floating-point rounding
          errors that can push the wrong token to the top and cascade into a
          repetition loop under greedy decoding.
        - A larger min_chunk_size on Windows lets APO noise suppression settle
          before committing, reducing false VAD triggers during pauses.
        """
        win_cpu = IS_WINDOWS and not use_gpu
        return _WhisperLiveKitConfig(  # type: ignore[misc]
            pcm_input=True, vac=True,
            model_size=size, lan=lan,
            decoder_type="beam" if (use_gpu or win_cpu) else "greedy",
            beams=2 if win_cpu else 1,
            min_chunk_size=0.40 if win_cpu else 0.30,
            audio_max_len=12.0, audio_min_len=0.20,
            direct_english_translation=False,
            static_init_prompt=prompt if prompt.strip() else None,
        )

    def _build_new_engine(self, size: str, lan: str, use_gpu: bool, prompt: str, my_gen: int) -> "object | None":
        """Construct a new TranscriptionEngine inside _exec_lock.

        Holds the lock for the full singleton reset + construction so that a
        zombie thread from a cancelled task cannot race a concurrent reload.
        Returns None if this generation has been superseded.
        """
        with self._exec_lock:
            if my_gen != self._load_gen:
                return None
            _TranscriptionEngine._instance    = None  # type: ignore[union-attr]
            _TranscriptionEngine._initialized = False  # type: ignore[union-attr]
            return _TranscriptionEngine(config=self._make_wlk_config(size, lan, use_gpu, prompt))  # type: ignore[misc]

    async def _try_load(
        self,
        model_size: str,
        fallback:   str,
        lan:        str,
        use_gpu:    bool,
        prompt:     str,
        my_gen:     int,
        lang:       str,
    ) -> "object | None":
        """Load model_size, falling back to fallback on GPU failure.

        Redirects stderr to capture tqdm download bars for the status bar.
        Returns the new engine, or None if superseded or permanently failed.
        """
        orig_stderr = None

        def maybe_capture_stderr(size: str) -> None:
            nonlocal orig_stderr
            if orig_stderr is None and not is_model_cached(size):
                orig_stderr = sys.stderr
                sys.stderr  = _TqdmCapture(self._on_status, orig_stderr)

        try:
            _log.debug("starting executor: build_new_engine size=%s gpu=%s", model_size, use_gpu)
            maybe_capture_stderr(model_size)
            engine = await asyncio.wait_for(
                self.loop.run_in_executor(  # type: ignore[union-attr]
                    None, lambda: self._build_new_engine(model_size, lan, use_gpu, prompt, my_gen),
                ),
                timeout=600.0,  # 10-minute cap; prevents frozen-exe deadlock hanging forever
            )
            _log.debug("executor finished: engine=%s", type(engine).__name__ if engine else None)
            if engine is None:
                return None  # superseded inside the executor
            return engine

        except Exception as exc:
            if my_gen != self._load_gen:
                return None  # superseded during download — silently discard

            can_fallback = use_gpu and model_size != fallback
            if not can_fallback:
                self._on_status(t("status.error", exc=exc))
                self._on_ready(False)
                return None

            # GPU primary failed — try CPU fallback model.
            self._on_status(t(
                "status.error_fallback",
                exc_type=exc.__class__.__name__,
                status=loading_status(fallback, lang, use_gpu),
            ))
            try:
                maybe_capture_stderr(fallback)
                engine = await asyncio.wait_for(
                    self.loop.run_in_executor(  # type: ignore[union-attr]
                        None, lambda: self._build_new_engine(fallback, lan, use_gpu, prompt, my_gen),
                    ),
                    timeout=600.0,
                )
                return engine  # None if superseded, engine otherwise

            except Exception as fallback_exc:
                if my_gen != self._load_gen:
                    return None
                self._on_status(t("status.error", exc=fallback_exc))
                self._on_ready(False)
                return None

        finally:
            if orig_stderr is not None:
                sys.stderr = orig_stderr

    async def _warmup_model(self, engine: object, model_size: str, lang: str, use_gpu: bool) -> None:
        """Run a dummy inference pass to pre-populate CTranslate2 caches.

        The first real inference after a cold load is 3–5× slower because the
        INT8 GEMM kernels are not yet JIT-compiled and the OS page-cache for
        model weights is cold.  Running noise through the model once brings both
        caches to steady-state so the user's first spoken word is processed at
        full speed.  Any exception is swallowed — warmup is best-effort.
        """
        if getattr(engine, "asr", None) is None:
            return
        device = "GPU" if use_gpu else "CPU"
        self._on_status(t("status.warmup", model=model_size, lang=lang, device=device))
        try:
            await asyncio.wait_for(
                self.loop.run_in_executor(  # type: ignore[union-attr]
                    None, lambda: engine.asr.transcribe(_make_warmup_audio()),  # type: ignore[union-attr]
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            _log.warning("Warmup timed out after 120 s — skipping warmup; first inference will be slower")
        except Exception:  # noqa: BLE001
            _log.warning("Warmup inference failed", exc_info=True)

    def _emit_ready_status(self, lang: str, use_gpu: bool) -> None:
        """Update the status bar with ready or AVX-512 warning, then fire on_ready."""
        if use_gpu or _has_avx512():
            self._on_status(t("status.ready", lang=lang))
        else:
            self._on_status(t("status.warn_no_avx512", lang=lang))
        self._on_ready(True)

    async def _load_engine(self, lang: str, prompt: str, speed: str = "normal") -> None:
        """Orchestrate model load: stamp the generation, acquire the lock, load,
        warm up, and signal the UI.  Aborts silently if superseded at any step."""
        from transcribe_app.config import get_model_size  # noqa: PLC0415

        use_gpu    = self.use_gpu
        opts       = LANGUAGE_OPTS[lang]
        model_size = get_model_size(lang, speed, use_gpu)
        fallback   = opts["fallback_model_size"]
        lan        = opts["lan"]

        # Stamp this request so any later reload can detect it has been superseded.
        self._load_gen += 1
        my_gen = self._load_gen

        # Update the status bar immediately so the UI reflects the new selection
        # even while waiting for the lock.
        self._on_status(loading_status(model_size, lang, use_gpu))

        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            # Re-check after acquiring: a later reload may have arrived
            # while we were waiting.
            if my_gen != self._load_gen:
                return

            self._lang = lang
            engine = await self._try_load(model_size, fallback, lan, use_gpu, prompt, my_gen, lang)
            if engine is None:
                return

            # Final check: another reload may have arrived while the executor ran.
            if my_gen != self._load_gen:
                return

            self._engine = engine
            await self._warmup_model(engine, model_size, lang, use_gpu)
            self._emit_ready_status(lang, use_gpu)

    # ── Session ───────────────────────────────────────────────────────────────

    async def _run_session(self) -> None:
        """Run one transcription session on the asyncio thread."""
        # Stamp this session so callbacks from a superseded session are silently
        # dropped even if the coroutine is still draining on the event loop.
        self._session_gen += 1
        my_gen = self._session_gen

        # Keep a local reference so the finally block always cleans up *this*
        # session's processor, not a replacement created by a concurrent restart.
        processor = _AudioProcessor(transcription_engine=self._engine)  # type: ignore[misc]
        self._processor = processor
        results_gen = await processor.create_tasks()
        # Signal the UI thread to open the mic stream via root.after(0, …)
        self._on_open_mic()

        committed_text = ""
        last_partial   = ""

        try:
            async for front_data in results_gen:
                # A new session was started (or this task was cancelled) — stop
                # updating the UI immediately so only one session drives the display.
                if my_gen != self._session_gen:
                    break

                # front_data.lines is the complete current snapshot of committed
                # segments — treat it as a replacement, not an accumulation.
                # Accumulating with a seen-set causes duplicates whenever
                # WhisperLiveKit refines a segment (text or timestamps change).
                new_committed = " ".join(
                    (seg.text or "").strip()
                    for seg in front_data.lines
                    if (seg.text or "").strip()
                )
                partial = (front_data.buffer_transcription or "").strip()

                if new_committed != committed_text or partial != last_partial:
                    committed_text = new_committed
                    last_partial   = partial
                    self._on_update(committed_text, partial)

            if my_gen == self._session_gen:
                final = f"{committed_text} {last_partial}".strip() if last_partial else committed_text
                self._on_finalise(final)

        finally:
            # EngineManager.stop_session() sends None to process_audio to flush;
            # here we just close the generator and release processor resources.
            for coro in (results_gen.aclose(), processor.cleanup()):
                try:
                    await coro
                except Exception:
                    pass
