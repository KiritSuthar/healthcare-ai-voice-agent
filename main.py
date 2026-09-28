"""Application entry point for the LiveKit voice agent."""

import logging

from dotenv import load_dotenv
from livekit import agents

from agent import create_server
from config import load_settings


def main() -> None:
    """Load local configuration and start the LiveKit worker CLI."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        settings = load_settings()
    except ValueError as error:
        raise SystemExit(str(error)) from error

    agents.cli.run_app(create_server(settings))


if __name__ == "__main__":
    main()