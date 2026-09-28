"""Isolated Opik tracing, transcript, tool, analysis, and evaluation adapter."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)


class OpikCallLogger:
    """Collect one call's telemetry and write it to a single Opik trace."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        project_name: str = "healthcare-voice-agent",
        enabled: bool = True,
    ) -> None:
        self._client = client
        self._project_name = project_name
        self._enabled = enabled and client is not None
        self._trace: Any | None = None
        self._call_start_time: datetime | None = None
        self._metadata: dict[str, Any] = {}
        self._transcript: list[dict[str, str]] = []
        self._tool_call_history: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._analysis: dict[str, Any] | None = None
        self._recording_reference = "unavailable"

    @classmethod
    def from_environment(cls) -> "OpikCallLogger":
        """Create an Opik client only when Cloud credentials or a self-host URL exist."""
        api_key = os.getenv("OPIK_API_KEY", "").strip()
        workspace = os.getenv("OPIK_WORKSPACE", "").strip()
        host = os.getenv("OPIK_URL_OVERRIDE", "").strip()
        project_name = os.getenv("OPIK_PROJECT_NAME", "healthcare-voice-agent").strip()

        if not host and not (api_key and workspace):
            logger.info("Opik is disabled: configure Opik credentials to enable call traces")
            return cls(project_name=project_name, enabled=False)

        try:
            from opik import Opik

            client = Opik(
                project_name=project_name,
                workspace=workspace or None,
                host=host or None,
                api_key=api_key or None,
            )
        except Exception:
            logger.warning("Could not initialize Opik; continuing without telemetry")
            return cls(project_name=project_name, enabled=False)

        return cls(client=client, project_name=project_name)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def transcript(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._transcript]

    @property
    def tool_call_history(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._tool_call_history]

    def start_trace(
        self,
        patient_metadata: Mapping[str, Any],
        *,
        call_start_time: datetime | None = None,
        recording_reference: str | None = None,
    ) -> None:
        """Start a trace and record the call's patient and initial call metadata."""
        self._call_start_time = call_start_time or datetime.now(timezone.utc)
        self._recording_reference = recording_reference or "unavailable"
        self._metadata = {
            "patient_name": patient_metadata.get("name"),
            "phone_number": patient_metadata.get("phone"),
            "blood_glucose": patient_metadata.get("blood_glucose"),
            "hba1c": patient_metadata.get("hba1c"),
            "call_start_time": self._call_start_time.isoformat(),
            "call_end_time": None,
            "call_duration_seconds": None,
            "call_recording_reference": self._recording_reference,
        }
        if not self._enabled:
            return

        try:
            self._trace = self._client.trace(
                name="healthcare_voice_call",
                start_time=self._call_start_time,
                input={"patient_metadata": dict(patient_metadata)},
                metadata=dict(self._metadata),
                project_name=self._project_name,
            )
        except Exception:
            logger.warning("Could not start Opik trace; continuing without telemetry")
            self._trace = None
            self._enabled = False

    def log_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Add call-level metadata; values are written when the trace is finished."""
        self._metadata.update(dict(metadata))

    def log_transcript(
        self,
        role: str,
        content: str,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Buffer one committed transcript message for the completed-call trace."""
        if not content.strip():
            return
        entry = {"role": role, "content": content}
        if timestamp is not None:
            entry["timestamp"] = timestamp.isoformat()
        self._transcript.append(entry)

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        timestamp: datetime | None = None,
    ) -> str:
        """Record the tool request and return an ID for its matching result."""
        call_id = str(uuid4())
        call_time = timestamp or datetime.now(timezone.utc)
        record = {
            "tool_name": tool_name,
            "tool_input": dict(tool_input),
            "tool_output": None,
            "timestamp": call_time.isoformat(),
            "result_timestamp": None,
        }
        self._tool_call_history.append(record)
        self._pending_tool_calls[call_id] = record
        return call_id

    def log_tool_result(
        self,
        call_id: str,
        tool_output: Any,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a tool result and add a completed tool span to the trace."""
        record = self._pending_tool_calls.pop(call_id, None)
        if record is None:
            logger.warning("Received a tool result without a matching tool call")
            return

        result_time = timestamp or datetime.now(timezone.utc)
        record["tool_output"] = tool_output
        record["result_timestamp"] = result_time.isoformat()
        if not self._enabled or self._trace is None:
            return

        try:
            self._trace.span(
                name=record["tool_name"],
                type="tool",
                input=record["tool_input"],
                output={"result": tool_output},
                start_time=datetime.fromisoformat(record["timestamp"]),
                end_time=result_time,
            )
        except Exception:
            logger.warning("Could not add tool span to Opik trace")

    def log_analysis(self, analysis: Mapping[str, Any]) -> None:
        """Attach post-call analysis to the final trace output."""
        self._analysis = dict(analysis)

    def run_evaluation(self, analysis: Mapping[str, Any]) -> dict[str, Any]:
        """Run and log the binary Goal Completion evaluation for this call."""
        completed = analysis.get("appointment_booked") is True
        score = 1.0 if completed else 0.0
        reason = (
            "Appointment booking was confirmed."
            if completed
            else "No confirmed appointment booking was recorded."
        )
        evaluation = {"name": "Goal Completion", "value": score, "reason": reason}

        if self._enabled and self._trace is not None:
            try:
                self._trace.log_feedback_score(
                    name=evaluation["name"],
                    value=score,
                    reason=reason,
                )
            except Exception:
                logger.warning("Could not attach Goal Completion score to Opik trace")
        return evaluation

    def finish_trace(
        self,
        *,
        call_end_time: datetime | None = None,
        recording_reference: str | None = None,
    ) -> None:
        """Finalize call timing/output and flush the trace to Opik."""
        end_time = call_end_time or datetime.now(timezone.utc)
        start_time = self._call_start_time or end_time
        duration_seconds = max(0.0, (end_time - start_time).total_seconds())
        if recording_reference:
            self._recording_reference = recording_reference
        self._metadata.update(
            {
                "call_end_time": end_time.isoformat(),
                "call_duration_seconds": duration_seconds,
                "call_recording_reference": self._recording_reference,
            }
        )

        if self._enabled and self._trace is not None:
            try:
                self._trace.end(
                    end_time=end_time,
                    metadata=dict(self._metadata),
                    output={
                        "conversation_transcript": self.transcript,
                        "tool_call_history": self.tool_call_history,
                        "post_call_analysis": self._analysis,
                        "call_recording_reference": self._recording_reference,
                    },
                )
                self._client.end(timeout=10)
            except Exception:
                logger.warning("Could not finish or flush Opik trace")
