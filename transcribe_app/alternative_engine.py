import sys
import time
import logging
import asyncio
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from queue import Queue, Full, Empty

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from scipy.signal import lfilter


_HP_B = np.array([1.0, -1.0], dtype=np.float64)   # high-pass numerator
_HP_A = np.array([1.0, -0.9975], dtype=np.float64) # high-pass denominator


LANGUAGE_TRANSLATION = {
    "Deutsch": "de",
    "English": "en"
}

# -----------------------------
# CONFIG
# -----------------------------
@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    frame_ms: int = 30
    input_device: int | None = None
    mic_gain: float = 1.0

    vad_aggressiveness: int = 2

    # Speech segmentation
    silence_limit: float = 1.0  # seconds of silence before flush

    # Buffer behavior
    max_buffer_seconds: float = 30.0  # safety cap
    min_buffer_chunks: int = 5        # avoid tiny transcriptions

    # Model
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"

    language: str = "auto"
    initial_prompt: str = ""

    queue_size: int = 200

    # Output mode: "file" | "stream" | "queue" | "print"
    #   file   — append each result to {output_prefix}.txt
    #   stream — write to output_stream (default: sys.stdout)
    #   queue  — put results into engine.result_queue for the caller to consume
    #   print  — call print() for each result
    output_mode: str = "stream"
    output_prefix: str = "session"  # used by "file" mode
    output_stream: object = None    # used by "stream" mode; None → sys.stdout

    # WAV snippet output: directory path for per-segment WAV files, or None to disable
    wav_snippet_dir: str | None = None


# -----------------------------
# LOGGER
# -----------------------------
log = logging.getLogger("asr")


# -----------------------------
# ENGINE
# -----------------------------
class SpeechToTextEngine:
    def __init__(self, cfg: Config, *, model=None, vad=None, status_cb=None):
        self.cfg = cfg

        self.frame_size = int(cfg.sample_rate * cfg.frame_ms / 1000)

        self.vad = vad if vad is not None else webrtcvad.Vad(cfg.vad_aggressiveness)
        self.model = model if model is not None else WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        self._status_cb = status_cb
        self._dc_zi        = np.zeros(1, dtype=np.float64)   # HP filter state
        self._clip_warn_t  = 0.0                              # throttle timestamp

        # thread-safe audio queue (callback → worker)
        self.audio_q: Queue = Queue(maxsize=cfg.queue_size)

        # optional sink called with each int16 frame from the sounddevice callback
        self.audio_sink = None  # Callable[[np.ndarray], None] | None

        # result queue — populated only in "queue" output mode
        self.result_queue: Queue = Queue()

        # resolved output stream for "stream" mode
        self._output_stream = cfg.output_stream if cfg.output_stream is not None else sys.stdout

        # buffer is ONLY used in worker thread (no locking needed)
        self.buffer = []

        # state (worker-owned → no race conditions)
        self.last_voice = time.time()
        self.session_id = 0
        self._snippet_counter = self._discover_next_snippet_id()

        self.stop_event  = Event()
        self.flush_done  = Event()  # set by worker after its final flush at stop time
        self.loop = asyncio.new_event_loop()

    def _discover_next_snippet_id(self) -> int:
        if not self.cfg.wav_snippet_dir:
            return 0
        out_dir = Path(self.cfg.wav_snippet_dir)
        try:
            ids = []
            for path in out_dir.glob("snippet_*.wav"):
                stem = path.stem
                try:
                    ids.append(int(stem.split("_")[-1]))
                except ValueError:
                    continue
            return (max(ids) + 1) if ids else 0
        except Exception:
            log.warning("Could not scan VAD snippet directory", exc_info=True)
            return 0

    # -------------------------
    # RESET
    # -------------------------
    def reset(self) -> None:
        """Prepare the engine for a fresh session without reloading the model."""
        self.stop_event.clear()
        self.flush_done.clear()
        self.buffer.clear()
        self.last_voice = time.time()
        self.session_id = 0
        for q in (self.audio_q, self.result_queue):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break
        # ProactorEventLoop (Windows) cannot be restarted after loop.stop().
        # Close the old loop and create a fresh one for the next session.
        if not self.loop.is_closed():
            self.loop.close()
        self.loop = asyncio.new_event_loop()

    # -------------------------
    # CALLBACK (ZERO LOGIC)
    # -------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning(status)

        samples = indata[:, 0].astype(np.float32, copy=False)
        gain = float(self.cfg.mic_gain)
        if gain != 1.0:
            samples = samples * gain
        samples = np.clip(samples, -1.0, 1.0)
        frame = np.rint(samples * 32767.0).astype(np.int16)

        if self.audio_sink is not None:
            self.audio_sink(frame)

        try:
            self.audio_q.put_nowait(frame)
        except Full:
            pass  # drop frame under backpressure

    # -------------------------
    # VAD
    # -------------------------
    def is_speech(self, frame: np.ndarray) -> bool:
        return self.vad.is_speech(frame.tobytes(), self.cfg.sample_rate)

    # -------------------------
    # WORKER THREAD
    # -------------------------
    def worker(self):
        log.info("worker started")

        while not self.stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.5)
            except Empty:
                continue

            now = time.time()
            speech = self.is_speech(frame)

            if speech:
                self.buffer.append(frame)
                self.last_voice = now
                continue

            # silence detection
            if self.buffer and (now - self.last_voice > self.cfg.silence_limit):
                self.flush()

        # Final flush: transcribe any buffered speech at stop time.
        # Run synchronously — the asyncio event loop may be shutting down
        # simultaneously, so we cannot use run_coroutine_threadsafe here.
        if self.buffer:
            audio = np.concatenate(self.buffer)
            self.buffer.clear()
            max_samples = int(self.cfg.max_buffer_seconds * self.cfg.sample_rate)
            if len(audio) > max_samples:
                audio = audio[-max_samples:]
            sid = self.session_id
            self.session_id += 1
            self._save_wav(audio, sid)
            try:
                text = self.transcribe_internal(audio.astype(np.float32) / 32768.0)
                if text.strip():
                    self._emit(text)
            except Exception:
                log.warning("Final flush transcription failed", exc_info=True)
        self.flush_done.set()

    # -------------------------
    # WAV SNIPPET SAVE
    # -------------------------
    def _save_wav(self, audio: np.ndarray, sid: int) -> None:
        if not self.cfg.wav_snippet_dir:
            return
        out_dir = Path(self.cfg.wav_snippet_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"snippet_{self._snippet_counter:04d}.wav"
            self._snippet_counter += 1
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes/sample
                wf.setframerate(self.cfg.sample_rate)
                wf.writeframes(audio.tobytes())
            log.debug("saved VAD snippet: %s", path)
        except Exception:
            log.warning("Failed to save VAD snippet %04d", sid, exc_info=True)

    # -------------------------
    # SAFE FLUSH (worker-owned)
    # -------------------------
    def flush(self):
        if len(self.buffer) < self.cfg.min_buffer_chunks:
            self.buffer.clear()
            return

        audio = np.concatenate(self.buffer)
        self.buffer.clear()

        # cap size
        max_samples = int(self.cfg.max_buffer_seconds * self.cfg.sample_rate)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]

        sid = self.session_id
        self.session_id += 1

        self._save_wav(audio, sid)

        asyncio.run_coroutine_threadsafe(
            self.transcribe(audio, sid),
            self.loop
        )

    # -------------------------
    # ASYNC TRANSCRIPTION
    # -------------------------

    def transcribe_internal(self, audio):
        language = None if self.cfg.language == "auto" else self.cfg.language
        initial_prompt = self.cfg.initial_prompt or None
        segments, _ = self.model.transcribe(
            audio,
            vad_filter=False,
            initial_prompt=initial_prompt,
            log_prob_threshold=-1.0,
            language=language,
        )
        return " ".join(s.text.strip() for s in segments)


    async def transcribe(self, audio: np.ndarray, sid: int):
        log.info(f"transcribing session {sid}")

        audio = audio.astype(np.float32) / 32768.0

        text = self.transcribe_internal(audio)

        if text.strip():
            log.info(f"session {sid}: {text[:80]}")
            self._emit(text)

    # -------------------------
    # OUTPUT DISPATCH
    # -------------------------
    def _emit(self, text: str):
        mode = self.cfg.output_mode
        if mode == "file":
            with open(f"{self.cfg.output_prefix}.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")
        elif mode == "stream":
            self._output_stream.write(text + "\n")
            self._output_stream.flush()
        elif mode == "queue":
            try:
                self.result_queue.put_nowait(text)
            except Full:
                log.warning("result_queue full — dropping result")
        elif mode == "print":
            print(text)
        else:
            raise ValueError(f"unknown output_mode: {mode!r}")

    # -------------------------
    # EVENT LOOP THREAD
    # -------------------------
    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -------------------------
    # RUN (internal, no I/O)
    # -------------------------
    def run_with_stream(self, stream_cm) -> None:
        """Run the engine using an already-opened audio stream context manager.

        Separated from ``run()`` so tests can supply a fake stream without
        touching sounddevice at all.
        """
        log.info("starting engine")

        loop_thread = Thread(target=self.start_loop, daemon=True)
        loop_thread.start()
        Thread(target=self.worker, daemon=True).start()

        with stream_cm:
            try:
                while not self.stop_event.is_set():
                    sd.sleep(500)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_event.set()
                self.loop.call_soon_threadsafe(self.loop.stop)
                loop_thread.join(timeout=5)
                log.info("engine stopped")

    # -------------------------
    # STREAM FACTORY
    # -------------------------
    def _make_stream(self) -> sd.InputStream:
        """Open the microphone InputStream.

        On Windows, tries WASAPI exclusive mode first to bypass the APO pipeline
        (noise suppression, echo cancellation, EQ) that makes recordings sound
        metallic.  Falls back to WASAPI shared mode if the device refuses exclusive
        access (e.g. another app already holds it).
        """
        kwargs = dict(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            latency="high",
            callback=self.audio_callback,
            device=self.cfg.input_device,
        )
        if sys.platform == "win32":
            try:
                return sd.InputStream(**kwargs, extra_settings=sd.WasapiSettings(exclusive=True))
            except Exception:
                log.warning(
                    "WASAPI exclusive mode unavailable; falling back to shared mode "
                    "(Windows audio enhancements may affect recording quality)"
                )
        return sd.InputStream(**kwargs)

    # -------------------------
    # RUN
    # -------------------------
    def run(self):
        self.run_with_stream(self._make_stream())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    config = Config(
        # language
        language="de",   # or "auto"

        # segmentation tuning
        silence_limit=0.5,
        min_buffer_chunks=10,

        # buffer safety
        max_buffer_seconds=30.0,

        # audio config
        sample_rate=16000,
        frame_ms=30,

        # VAD
        vad_aggressiveness=2,

        # model
        model_size="base",
        device="cpu",
        compute_type="int8",

        # output
        output_mode="stream",   # "file" | "stream" | "queue" | "print"
        output_prefix="session",

        # queue safety
        queue_size=200,
    )

    engine = SpeechToTextEngine(config)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
