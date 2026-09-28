"""Non-streaming LiveKit STT adapter backed by local faster-whisper."""

import asyncio
import logging
import os
import sysconfig
import threading
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils.audio import AudioBuffer

logger = logging.getLogger(__name__)
WHISPER_SAMPLE_RATE = 16_000


def _configure_cuda_dll_search() -> None:
    """Expose pip-installed NVIDIA DLL folders to Windows' loader."""
    if os.name != "nt":
        return

    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return

    for bin_dir in nvidia_root.glob("*/bin"):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(bin_dir))
        except (FileNotFoundError, OSError):
            logger.debug("Could not register CUDA DLL directory: %s", bin_dir)


_configure_cuda_dll_search()
import ctranslate2  # noqa: E402


class FasterWhisperSTT(stt.STT):
    """Run one local faster-whisper transcription for each completed utterance."""

    def __init__(self, model_size: str = "tiny.en", language: str = "en") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model_size = model_size
        self._language = language
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()
        self._using_cuda = False

    @property
    def model(self) -> str:
        return self._model_size

    @property
    def provider(self) -> str:
        return "faster-whisper-local"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        requested_language = (
            language if isinstance(language, str) else self._language or None
        )
        try:
            transcript, detected_language = await asyncio.to_thread(
                self._transcribe,
                buffer,
                requested_language,
            )
        except Exception as error:
            logger.exception("faster-whisper failed to transcribe an utterance")
            raise RuntimeError("Local faster-whisper transcription failed") from error

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language=detected_language or requested_language or "en",
                    text=transcript,
                )
            ],
        )

    def _transcribe(
        self, audio_buffer: AudioBuffer, language: str | None
    ) -> tuple[str, str | None]:
        """Convert room audio and run the blocking model off the event loop."""
        audio = self._to_model_audio(audio_buffer)
        if audio.size == 0:
            return "", language

        model = self._get_model()
        try:
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return transcript, info.language
        except RuntimeError as error:
            if not self._using_cuda or "cublas" not in str(error).lower():
                raise

            # CTranslate2 can report CUDA as available and fail later when it loads cuBLAS.
            # Retry once on CPU so a missing CUDA runtime does not break the conversation.
            logger.warning(
                "CUDA runtime is unavailable; retrying faster-whisper on CPU"
            )
            self._model = self._load_model(device="cpu", compute_type="int8")
            self._using_cuda = False
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
            )

        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return transcript, info.language

    def _get_model(self) -> WhisperModel:
        """Load the model once, preferring CUDA and falling back to CPU int8."""
        with self._model_lock:
            if self._model is not None:
                return self._model

            if ctranslate2.get_cuda_device_count() > 0:
                try:
                    self._model = self._load_model(device="cuda", compute_type="float16")
                    self._using_cuda = True
                    logger.info("Loaded faster-whisper model on CUDA")
                    return self._model
                except Exception:
                    logger.warning(
                        "CUDA initialization failed; loading faster-whisper on CPU",
                        exc_info=True,
                    )

            try:
                self._model = self._load_model(device="cpu", compute_type="int8")
                self._using_cuda = False
            except Exception as error:
                raise RuntimeError(
                    f"Could not load faster-whisper model '{self._model_size}'"
                ) from error

            logger.info("Loaded faster-whisper model on CPU with int8")
            return self._model

    def _load_model(self, *, device: str, compute_type: str) -> WhisperModel:
        """Construct a faster-whisper model with the requested compute backend."""
        return WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )

    def prewarm(self) -> None:
        """Load the model before the first user turn."""
        self._get_model()

    @staticmethod
    def _to_model_audio(audio_buffer: AudioBuffer) -> np.ndarray:
        """Downmix frames and resample them to faster-whisper's 16 kHz input."""
        frames = audio_buffer if isinstance(audio_buffer, list) else [audio_buffer]
        if not frames:
            return np.empty(0, dtype=np.float32)

        sample_rate = frames[0].sample_rate
        if any(frame.sample_rate != sample_rate for frame in frames):
            raise ValueError("Audio frames in one utterance must share a sample rate")

        mono_chunks: list[np.ndarray] = []
        for frame in frames:
            samples = np.frombuffer(frame.data, dtype=np.int16)
            samples = samples.reshape(frame.samples_per_channel, frame.num_channels)
            mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype(np.int16)
            mono_chunks.append(mono)

        mono_samples = np.concatenate(mono_chunks)
        if sample_rate == WHISPER_SAMPLE_RATE:
            resampled = mono_samples
        else:
            mono_frame = rtc.AudioFrame(
                data=mono_samples.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(mono_samples),
            )
            resampler = rtc.AudioResampler(
                input_rate=sample_rate,
                output_rate=WHISPER_SAMPLE_RATE,
                num_channels=1,
            )
            output_frames = resampler.push(mono_frame)
            output_frames.extend(resampler.flush())
            if not output_frames:
                return np.empty(0, dtype=np.float32)
            resampled = np.concatenate(
                [
                    np.frombuffer(frame.data, dtype=np.int16)
                    for frame in output_frames
                ]
            )

        return resampled.astype(np.float32) / 32768.0