# Healthcare Voice Agent

## Project Overview

The project uses the existing LiveKit voice pipeline for a patient-aware healthcare conversation. Phase 3 adds post-call analysis, isolated Opik tracing, and a deterministic Goal Completion score. Gemini remains supported; Ollama can run live replies and post-call analysis locally. The booking tool remains simulated.

## Architecture

```text
LiveKit room/job
  -> agent.py (existing voice session + event capture)
   -> faster-whisper STT -> healthcare Agent/selected LLM -> Kokoro TTS
  -> schedule_appointment tool (existing simulated tool)
  -> job shutdown callback
      -> analysis.py (provider-selected JSON analysis)
       -> opik_logger.py (one call trace, transcript, tools, analysis, score)
```

`main.py`, `config.py`, the healthcare prompt, patient model, appointment logic, and audio providers are unchanged. `agent.py` only adds lifecycle/event wiring. Opik SDK calls are contained in `opik_logger.py`; missing Opik configuration or telemetry failures do not fail a call.

## Installation

Activate the existing project environment and install the updated dependencies:

```cmd
cd /d C:\Assesment\Assesment_adit
venv\Scripts\activate
python -m pip install -r requirements.txt
```

The existing Phase 1 prerequisites still apply: LiveKit credentials, Gemini API key, and a reachable Kokoro-FastAPI service.

## Environment Variables

Keep the existing Phase 1 settings in `.env`. To keep Gemini, omit `LLM_PROVIDER` or set it to `gemini`. To use a local Ollama model for both live turns and post-call analysis, configure:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_API_KEY=ollama
```

Ollama runs locally and does not require a Google API key when `LLM_PROVIDER=ollama`. Its service must be running before the agent handles a call. Gemini remains the default and still requires `GOOGLE_API_KEY`.

Add these Opik settings for Opik Cloud:

```env
OPIK_API_KEY=your-opik-api-key
OPIK_WORKSPACE=your-opik-workspace
OPIK_PROJECT_NAME=healthcare-voice-agent
```

For a self-hosted Opik server, set `OPIK_URL_OVERRIDE` to its API URL instead; Opik credentials are optional when the server does not require them. If Opik configuration is absent, voice calls continue without Opik traces.

## How to Run

1. Start Kokoro-FastAPI in its own terminal.
2. For local inference, make sure Ollama is running and the selected model is available:

   ```cmd
   ollama pull qwen2.5:3b
   ```

   The default Ollama-compatible URL is `http://localhost:11434/v1`.
3. Start the LiveKit agent:

   ```cmd
   cd /d C:\Assesment\Assesment_adit
   venv\Scripts\activate
   python main.py dev
   ```

4. Dispatch `voice-assistant` with patient JSON in LiveKit job metadata, then join the room and complete a test conversation.
5. End the session. The LiveKit job shutdown callback runs post-call analysis and writes the completed trace/evaluation.

Use synthetic patient values for demos. The assignment's trace includes patient name, phone, biomarker values, and transcript text; only send real patient information to an Opik deployment/workspace that has been approved for the applicable privacy, access-control, and retention requirements.

## Complete Call Flow

1. The agent reads and validates patient JSON from job metadata.
2. Existing STT, healthcare conversation, tool-calling, and TTS behavior run without modification.
3. LiveKit conversation events buffer user/assistant transcript messages. Tool events buffer each tool name, input, output, and timestamp.
4. At job shutdown, `analysis.py` sends patient metadata, the complete transcript, tool history, and booking result to the selected LLM provider and validates its JSON response. Analysis failure is recorded as unavailable and does not interrupt call shutdown.
5. `opik_logger.py` finalizes call metadata (start/end/duration), transcript, tool spans, analysis, and a recording reference. Since LiveKit recording is disabled in this project, the reference is `unavailable` unless one is supplied.

Post-call analysis makes one additional request to the selected LLM provider per call. When using Ollama, the call analysis runs locally and does not consume Gemini API quota.

## Opik Integration

All Opik SDK usage lives in `opik_logger.py`. It exposes `start_trace`, `log_metadata`, `log_transcript`, `log_tool_call`, `log_tool_result`, `log_analysis`, `finish_trace`, and `run_evaluation`. Opik stores a single call trace with patient metadata, transcript, tool spans/history, analysis JSON, call timing, and the recording placeholder/reference.

## Evaluation

After the analysis attempt, `run_evaluation()` scores **Goal Completion** from the authoritative appointment tool result: `1` when its status is `confirmed`, otherwise `0`. The score and reason are attached to the call trace as an Opik feedback score. This is a deterministic per-call evaluation; it does not make another LLM call.