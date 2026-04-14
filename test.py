import time
import logging
import asyncio
from dataclasses import dataclass
from threading import Event, Lock, Thread
from queue import Queue, Full

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


# -----------------------------
# CONFIG
# -----------------------------
@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    frame_ms: int = 30

    vad_aggressiveness: int = 2

    # Speech segmentation
    silence_limit: float = 1.0  # seconds of silence before flush

    # Buffer behavior
    max_buffer_seconds: float = 30.0  # safety cap
    min_buffer_chunks: int = 5        # avoid tiny transcriptions

    # Model
    model_size: str = "base"
    device: str = "cuda"
    compute_type: str = "float16"

    language: str = "auto"

    queue_size: int = 200

    file_mode: str = "append"
    output_prefix: str = "session"


# -----------------------------
# LOGGER
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("asr")


# -----------------------------
# ENGINE
# -----------------------------
class SpeechToTextEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.frame_size = int(cfg.sample_rate * cfg.frame_ms / 1000)

        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type
        )

        # 🔒 thread-safe audio queue (callback → worker)
        self.audio_q: Queue = Queue(maxsize=cfg.queue_size)

        # buffer is ONLY used in worker thread
        self.buffer = []

        # state (worker-owned → no race conditions)
        self.last_voice = time.time()
        self.session_id = 0

        self.stop_event = Event()
        self.lock = Lock()

        self.loop = asyncio.new_event_loop()

    # -------------------------
    # CALLBACK (ZERO LOGIC)
    # -------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning(status)

        frame = (indata[:, 0] * 32768).astype(np.int16)

        try:
            self.audio_q.put_nowait(frame)
        except Full:
            # 🔥 backpressure safety (drop frame)
            pass

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
            except:
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

        asyncio.run_coroutine_threadsafe(
            self.transcribe(audio, sid),
            self.loop
        )

    # -------------------------
    # ASYNC TRANSCRIPTION (SAFE)
    # -------------------------
    async def transcribe(self, audio: np.ndarray, sid: int):
        log.info(f"🧠 session {sid}")

        audio = audio.astype(np.float32) / 32768.0
        language = None if self.cfg.language == "auto" else self.cfg.language

        def _run():
            segments, _ = self.model.transcribe(
                audio,
                vad_filter=False,
                #beam_size=2,
                initial_prompt="Sirve",
                log_prob_threshold=-1.0,
                language=language
            )
            return " ".join(s.text.strip() for s in segments)

        text = await asyncio.to_thread(_run)

        if text.strip():
            log.info(f"📝 {text}")
            with open(f"{self.cfg.output_prefix}_{sid}.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")

    # -------------------------
    # EVENT LOOP THREAD
    # -------------------------
    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -------------------------
    # RUN
    # -------------------------
    def run(self):
        log.info("🎤 starting engine")

        Thread(target=self.start_loop, daemon=True).start()
        Thread(target=self.worker, daemon=True).start()

        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            latency="high",
            callback=self.audio_callback,
        ):
            try:
                while not self.stop_event.is_set():
                    sd.sleep(500)
            except KeyboardInterrupt:
                self.stop_event.set()
                log.info("stopping engine")

if __name__ == "__main__":
    config = Config(
        # 🌍 language
        language="de",   # or "auto"

        # ⏱ segmentation tuning
        silence_limit=0.5,
        min_buffer_chunks=10,

        # 🧠 buffer safety
        max_buffer_seconds=30.0,

        # 🎧 audio config
        sample_rate=16000,
        frame_ms=30,

        # ⚙️ VAD
        vad_aggressiveness=2,

        # 🤖 model
        model_size="base",
        device="cpu",            # fallback to "cpu" if needed
        compute_type="int8",   # change to "int8_float16" if CUDA errors

        # 💾 output
        file_mode="append",
        output_prefix="session",

        # 🔥 queue safety
        queue_size=200,
    )

    engine = SpeechToTextEngine(config)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")