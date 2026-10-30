"""All settings in one place, read from the .env file."""

import os

from dotenv import load_dotenv

# Reads .env and puts its values into environment variables.
load_dotenv()


def _required(name: str) -> str:
    """Return a setting, or stop with a clear message if it's missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing setting {name}. Add it to your .env file (see .env.example).")
    return value


OPENAI_API_KEY = _required("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
