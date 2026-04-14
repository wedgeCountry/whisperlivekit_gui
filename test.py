import time
import logging
from dataclasses import dataclass
from collections import deque
from threading import Event

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


# -----------------------------
# CONFIGURATION
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Audio
    sample_rate: int = 16000
    frame_ms: int = 30

    # VAD
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

    # Language
    language: str = "auto"

    # 💾 File writing config
    file_mode: str = "append"   # "append" | "overwrite"
    write_each_session: bool = True
    output_prefix: str = "session"

    # 🧠 Buffer writing control
    write_empty_sessions: bool = False


# -----------------------------
# LOGGER
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("speech-to-text")


# -----------------------------
# ENGINE
# -----------------------------
class SpeechToTextEngine:
    def __init__(self, config: Config):
        self.config = config

        self.frame_size = int(config.sample_rate * config.frame_ms / 1000)

        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type
        )

        self.buffer = deque()
        self.last_speech_time = None
        self.session_id = 0

        self.stop_event = Event()

    # -------------------------
    # FILE WRITING
    # -------------------------
    def write_session(self, text: str, sid: int) -> None:
        if not text and not self.config.write_empty_sessions:
            return

        filename = f"{self.config.output_prefix}_{sid}.txt"

        mode = "w" if self.config.file_mode == "overwrite" else "a"

        with open(filename, mode, encoding="utf-8") as f:
            f.write(text.strip() + "\n")

        logger.info(f"💾 Wrote session {sid} ({mode})")

    # -------------------------
    # VAD
    # -------------------------
    def is_speech(self, frame: np.ndarray) -> bool:
        try:
            return self.vad.is_speech(frame.tobytes(), self.config.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    # -------------------------
    # AUDIO CALLBACK
    # -------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(status)

        if self.stop_event.is_set():
            return

        try:
            frame = (indata[:, 0] * 32768).astype(np.int16)

            if len(frame) != self.frame_size:
                return

            now = time.time()
            speech = self.is_speech(frame)

            if speech:
                self.buffer.append(frame)
                self.last_speech_time = now
                return

            # -------------------------
            # SILENCE DETECTION
            # -------------------------
            if (
                self.last_speech_time
                and (now - self.last_speech_time > self.config.silence_limit)
            ):
                self.flush_buffer()

        except Exception as e:
            logger.exception(f"Audio callback error: {e}")

    # -------------------------
    # BUFFER MANAGEMENT
    # -------------------------
    def flush_buffer(self):
        if not self.buffer:
            return

        if len(self.buffer) < self.config.min_buffer_chunks:
            logger.info("⏭ Skipping short buffer")
            self.buffer.clear()
            return

        audio = np.concatenate(list(self.buffer))
        self.buffer.clear()

        # safety cap
        max_samples = int(self.config.max_buffer_seconds * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]

        self.process_audio(audio, self.session_id)
        self.session_id += 1

    # -------------------------
    # TRANSCRIPTION
    # -------------------------
    def process_audio(self, audio: np.ndarray, sid: int):
        logger.info(f"🧠 Transcribing session {sid}...")

        audio = audio.astype(np.float32) / 32768.0

        language = None if self.config.language == "auto" else self.config.language

        segments, _ = self.model.transcribe(
            audio,
            vad_filter=False,
            language=language
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if text:
            logger.info(f"📝 RESULT: {text}")
            self.write_session(text, sid)

    # -------------------------
    # RUN LOOP
    # -------------------------
    def run(self):
        logger.info("🎤 Engine started")
        logger.info(f"🌐 Language: {self.config.language}")
        logger.info(f"⏱ Silence limit: {self.config.silence_limit}s")
        logger.info(f"💾 File mode: {self.config.file_mode}")

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                callback=self.audio_callback,
            ):
                while not self.stop_event.is_set():
                    sd.sleep(500)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")

        finally:
            self.stop_event.set()
            logger.info("Engine stopped cleanly")


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    config = Config(
        language="de", # "auto"
        silence_limit=0.6,          # ⏱ tweak silence detection
        file_mode="append",         # 💾 append or overwrite
        min_buffer_chunks=6,        # 🧠 require more speech before processing
        write_each_session=True
    )

    engine = SpeechToTextEngine(config)
    engine.run()