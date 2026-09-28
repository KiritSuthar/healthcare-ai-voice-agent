"""LiveKit session setup and assistant behavior."""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    TurnHandlingOptions,
    function_tool,
)
from livekit.plugins import google, openai
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    ToolExecutionUpdatedEvent,
)

from analysis import analyze_call
from config import Settings
from faster_whisper_stt import FasterWhisperSTT
from opik_logger import OpikCallLogger
from patient import Patient
from prompts import build_healthcare_prompt, build_opening
from tools import schedule_appointment as book_appointment

logger = logging.getLogger(__name__)


def _decode_tool_payload(value: str) -> Any:
    """Decode JSON-encoded LiveKit tool arguments or output when possible."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


class HealthcareAssistant(Agent):
    """Patient-aware assistant with one simulated booking tool."""

    def __init__(self, patient: Patient) -> None:
        self.patient = patient
        super().__init__(instructions=build_healthcare_prompt(patient))

    @function_tool(
        description=(
            "Book a doctor consultation after the patient has clearly agreed to schedule one. "
            "Do not call this tool before explicit agreement."
        )
    )
    async def schedule_appointment(self, context: RunContext) -> dict[str, str]:
        """Simulate a consultation booking for this call's patient."""
        try:
            return book_appointment(self.patient)
        except (TypeError, ValueError) as error:
            logger.exception("Appointment simulation failed")
            raise RuntimeError("Unable to schedule the consultation") from error


def create_server(settings: Settings) -> AgentServer:
    """Build the LiveKit server and its per-room voice pipeline."""
    server = AgentServer()

    @server.rtc_session(agent_name="voice-assistant")
    async def voice_session(ctx: agents.JobContext) -> None:
        # Each call receives its patient record in LiveKit dispatch/job metadata.
        try:
            patient = Patient.from_metadata(ctx.job.metadata)
        except ValueError:
            logger.exception("Voice session has missing or invalid patient metadata")
            raise

        call_start_time = datetime.now(timezone.utc)
        call_logger = OpikCallLogger.from_environment()
        call_logger.start_trace(asdict(patient), call_start_time=call_start_time)
        appointment_booking_result: dict[str, Any] | None = None
        tool_log_ids: dict[str, str] = {}
        session_started = False
        session: AgentSession | None = None

        def on_conversation_item(event: ConversationItemAddedEvent) -> None:
            item = event.item
            role = getattr(item, "role", None)
            text = getattr(item, "raw_text_content", None)
            if role in {"user", "assistant"} and isinstance(text, str) and text.strip():
                call_logger.log_transcript(
                    role,
                    text,
                    timestamp=datetime.fromtimestamp(event.created_at, timezone.utc),
                )

        def on_tool_started(event: ToolExecutionUpdatedEvent) -> None:
            update = event.update
            if update.type != "tool_call_started":
                return

            function_call = update.function_call
            arguments = _decode_tool_payload(function_call.arguments)
            tool_input = arguments if isinstance(arguments, dict) else {"arguments": arguments}
            tool_log_ids[function_call.call_id] = call_logger.log_tool_call(
                function_call.name,
                tool_input,
                timestamp=datetime.fromtimestamp(event.created_at, timezone.utc),
            )

        def on_tools_executed(event: FunctionToolsExecutedEvent) -> None:
            nonlocal appointment_booking_result
            for function_call, function_output in event.zipped():
                log_id = tool_log_ids.get(function_call.call_id)
                if log_id is None:
                    arguments = _decode_tool_payload(function_call.arguments)
                    tool_input = (
                        arguments if isinstance(arguments, dict) else {"arguments": arguments}
                    )
                    log_id = call_logger.log_tool_call(function_call.name, tool_input)

                output = _decode_tool_payload(function_output.output)
                call_logger.log_tool_result(log_id, output)
                if function_call.name == "schedule_appointment":
                    if not function_output.is_error and isinstance(output, dict):
                        appointment_booking_result = output
                    else:
                        appointment_booking_result = {
                            "status": "failed",
                            "error": str(output),
                        }

        async def finalize_call(shutdown_reason: str) -> None:
            """Analyze and log the call once LiveKit has shut down the session."""
            if session_started and session is not None:
                try:
                    await session.aclose()
                except Exception:
                    logger.warning("Could not close the voice session before analysis")

            call_end_time = datetime.now(timezone.utc)
            call_logger.log_metadata({"shutdown_reason": shutdown_reason})
            if not session_started:
                analysis_result = {
                    "appointment_booked": False,
                    "patient_interest": "unknown",
                    "summary": "The voice session did not start.",
                    "follow_up_required": False,
                }
            else:
                try:
                    analysis_result = await analyze_call(
                        patient_metadata=asdict(patient),
                        conversation_transcript=call_logger.transcript,
                        tool_call_history=call_logger.tool_call_history,
                        appointment_booking_result=appointment_booking_result,
                        provider=settings.llm_provider,
                        api_key=settings.google_api_key,
                        model=(
                            settings.ollama_model
                            if settings.llm_provider == "ollama"
                            else settings.gemini_model
                        ),
                        base_url=(
                            settings.ollama_base_url
                            if settings.llm_provider == "ollama"
                            else None
                        ),
                    )
                except Exception as error:
                    logger.exception("Post-call Gemini analysis failed")
                    analysis_result = {
                        "appointment_booked": bool(
                            appointment_booking_result
                            and appointment_booking_result.get("status") == "confirmed"
                        ),
                        "patient_interest": "unknown",
                        "summary": "Post-call analysis was unavailable.",
                        "follow_up_required": False,
                    }
                    call_logger.log_metadata(
                        {"analysis_status": "failed", "analysis_error": str(error)}
                    )
                else:
                    call_logger.log_metadata({"analysis_status": "completed"})

            call_logger.log_analysis(analysis_result)
            call_logger.run_evaluation(analysis_result)
            call_logger.finish_trace(call_end_time=call_end_time)

        # Register before model/session startup so failed jobs still finalize their trace.
        ctx.add_shutdown_callback(finalize_call)

        # Each room gets its own local STT instance and LiveKit session.
        stt = FasterWhisperSTT(
            model_size=settings.whisper_model,
            language=settings.whisper_language,
        )
        # Model loading is blocking; move it off the event loop before accepting speech.
        await asyncio.to_thread(stt.prewarm)

        session = AgentSession(
            stt=stt,
            llm=(
                openai.LLM(
                    model=settings.ollama_model,
                    api_key=settings.ollama_api_key,
                    base_url=settings.ollama_base_url,
                    temperature=0.1,
                    timeout=120.0,
                    max_completion_tokens=160,
                )
                if settings.llm_provider == "ollama"
                else google.LLM(model=settings.gemini_model)
            ),
            tts=openai.TTS(
                model="kokoro",
                voice=settings.kokoro_voice,
                api_key=settings.kokoro_api_key,
                base_url=settings.kokoro_base_url,
                response_format="wav",
            ),
            turn_handling=TurnHandlingOptions(
                turn_detection="vad",
                endpointing={
                    "min_delay": 0.35,
                    "max_delay": 1.5,
                },
                preemptive_generation={"enabled": False},
            ),
        )
        session.on("conversation_item_added", on_conversation_item)
        session.on("tool_execution_updated", on_tool_started)
        session.on("function_tools_executed", on_tools_executed)

        try:
            # Starting with the job room connects room audio to the voice pipeline.
            await session.start(
                room=ctx.room,
                agent=HealthcareAssistant(patient),
                record=False,
            )
            session_started = True

            # Say the opening directly; conversation and tool decisions use the selected LLM.
            greeting = session.say(build_opening(patient))
            await greeting
            if greeting.exception() is not None:
                raise RuntimeError("The initial greeting could not be spoken") from greeting.exception()
        except Exception:
            logger.exception("Voice session failed for room %s", ctx.room.name)
            raise

    return server