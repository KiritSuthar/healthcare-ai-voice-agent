"""Dispatch a healthcare voice-agent test call with synthetic patient data."""

import asyncio
import json
import os
from datetime import timedelta
from uuid import uuid4

from dotenv import load_dotenv
from livekit import api


async def main() -> None:
    """Create a LiveKit agent dispatch and print the room to join."""
    load_dotenv()

    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise SystemExit("Missing .env values: " + ", ".join(missing))

    # Use synthetic values only; do not test with real patient information.
    patient_metadata = {
        "name": "Kirit Suthar",
        "phone": "+918696888617",
        "blood_glucose": 185,
        "hba1c": 9.3,
    }
    room_name = f"phase2-test-{uuid4().hex[:8]}"
    livekit_url = os.environ["LIVEKIT_URL"].strip()
    api_key = os.environ["LIVEKIT_API_KEY"].strip()
    api_secret = os.environ["LIVEKIT_API_SECRET"].strip()

    # This temporary participant token can join and publish audio only in this test room.
    participant_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(f"test-participant-{uuid4().hex[:8]}")
        .with_name("Phase 2 Test Participant")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(timedelta(minutes=15))
        .to_jwt()
    )

    client = api.LiveKitAPI(
        url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
    )
    try:
        await client.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="voice-assistant",
                room=room_name,
                metadata=json.dumps(patient_metadata),
            )
        )
    finally:
        await client.aclose()

    print(f"LiveKit Server URL: {livekit_url}")
    print(f"Room Name: {room_name}")
    print("Paste this temporary 15-minute participant token into LiveKit Meet:")
    print(participant_token)
    print("Keep the token private; it only grants access to this test room.")


if __name__ == "__main__":
    asyncio.run(main())
