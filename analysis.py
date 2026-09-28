"""Post-call conversation analysis using structured JSON model output."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from google import genai
from google.genai import types
from openai import AsyncOpenAI


class CallAnalysis(TypedDict):
    """Validated fields returned by post-call analysis."""

    appointment_booked: bool
    patient_interest: Literal["high", "medium", "low", "unknown"]
    summary: str
    follow_up_required: bool


async def analyze_call(
    patient_metadata: Mapping[str, Any],
    conversation_transcript: str | Sequence[Mapping[str, str]],
    tool_call_history: Sequence[Mapping[str, Any]],
    appointment_booking_result: Mapping[str, Any] | None,
    *,
    provider: Literal["gemini", "ollama"] = "gemini",
    api_key: str = "",
    model: str,
    base_url: str | None = None,
) -> CallAnalysis:
    """Analyze a completed call and return a validated JSON-compatible result.

    The appointment result is treated as authoritative instead of relying on the
    model to infer whether a booking succeeded.
    """
    if provider == "gemini" and not api_key.strip():
        raise ValueError("A Gemini API key is required for call analysis")
    if provider not in {"gemini", "ollama"}:
        raise ValueError("Analysis provider must be 'gemini' or 'ollama'")

    transcript = (
        conversation_transcript
        if isinstance(conversation_transcript, str)
        else "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in conversation_transcript
        )
    )
    booked = bool(
        appointment_booking_result
        and appointment_booking_result.get("status") == "confirmed"
    )

    prompt = """Analyze this completed healthcare voice-agent call.
Summarize the conversation, estimate patient interest as high, medium, low, or unknown, and determine whether follow-up is required.
Do not diagnose, prescribe, add medical claims, or invent patient facts. Treat the appointment tool result as the only source of truth for booking status. Return only a JSON object with exactly these fields: appointment_booked (boolean), patient_interest (one of high/medium/low/unknown), summary (string), follow_up_required (boolean).

Patient metadata:
{patient}

Conversation transcript:
{transcript}

Tool call history:
{tool_history}

Appointment booking result:
{booking_result}
""".format(
        patient=json.dumps(dict(patient_metadata), ensure_ascii=True),
        transcript=transcript,
        tool_history=json.dumps(list(tool_call_history), ensure_ascii=True),
        booking_result=json.dumps(
            dict(appointment_booking_result) if appointment_booking_result else None,
            ensure_ascii=True,
        ),
    )

    if provider == "ollama":
        client = AsyncOpenAI(
            api_key=api_key or "ollama",
            base_url=base_url,
            timeout=120.0,
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=300,
            )
        finally:
            await client.close()
        response_text = response.choices[0].message.content or ""
    else:
        client = genai.Client(api_key=api_key)
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        finally:
            await client.aio.aclose()
            client.close()
        response_text = response.text or ""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini returned invalid JSON for call analysis") from error

    if not isinstance(payload, dict):
        raise ValueError("Gemini call analysis must be a JSON object")

    interest = payload.get("patient_interest")
    if interest not in {"high", "medium", "low", "unknown"}:
        interest = "unknown"
    summary = payload.get("summary")
    follow_up = payload.get("follow_up_required")
    if not isinstance(summary, str) or not isinstance(follow_up, bool):
        raise ValueError("Gemini call analysis is missing valid required fields")

    return {
        "appointment_booked": booked,
        "patient_interest": interest,
        "summary": summary,
        "follow_up_required": follow_up,
    }
