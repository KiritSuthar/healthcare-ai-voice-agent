"""Environment-based application configuration."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings without embedding credentials in source code."""

    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    google_api_key: str
    llm_provider: str
    gemini_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_api_key: str
    whisper_language: str
    kokoro_base_url: str
    kokoro_api_key: str
    kokoro_voice: str
    whisper_model: str


def load_settings() -> Settings:
    """Read required credentials and optional model/service settings."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "ollama"}:
        raise ValueError("LLM_PROVIDER must be either 'gemini' or 'ollama'")

    required_names = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]
    if provider == "gemini":
        required_names.append("GOOGLE_API_KEY")
    missing = [name for name in required_names if not os.getenv(name, "").strip()]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return Settings(
        livekit_url=os.environ["LIVEKIT_URL"].strip(),
        livekit_api_key=os.environ["LIVEKIT_API_KEY"].strip(),
        livekit_api_secret=os.environ["LIVEKIT_API_SECRET"].strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        llm_provider=provider,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        ).strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip(),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", "ollama").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        kokoro_base_url=os.getenv(
            "KOKORO_BASE_URL", "http://localhost:8880/v1"
        ).strip(),
        # Kokoro-FastAPI does not require authentication; the OpenAI client needs a non-empty value.
        kokoro_api_key=os.getenv("KOKORO_API_KEY", "not-needed").strip(),
        kokoro_voice=os.getenv("KOKORO_VOICE", "af_sky").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "tiny.en").strip(),
    )