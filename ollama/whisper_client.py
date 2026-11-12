"""Client for the mlx-whisper child process (whisper_server.py).

The child is on loopback, started and stopped by `ModelManager`, and speaks two
calls: `/health` and `/transcribe`. This file is the gateway's half.

Both a sync and an async method, on purpose:

  - `health()` is sync because its caller is `load_whisper_model`, which is a
    blocking spawn-and-poll loop run off the event loop with `asyncio.to_thread`.
  - `transcribe()` is async because its caller is a request handler, and a
    transcription runs for minutes. A blocking httpx call there would freeze
    every other route for the length of the audio — the same failure the pool's
    `/state` read caused before it was moved to a thread.
"""

import httpx
from typing import Optional

from . import config


class WhisperUnavailable(Exception):
    """The child is not answering; this alone does not establish backend end."""


class WhisperFailed(Exception):
    """The child answered with a refusal. Carries its status and detail.

    `status` is the child's, and the gateway maps it rather than forwarding it:
    the child's 422 ("ffmpeg could not decode this") is a 400 to the caller,
    whose request is what was wrong.
    """

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class WhisperClient:
    def __init__(
        self,
        host: str = config.WHISPER_HOST,
        port: int = config.WHISPER_PORT,
        timeout: int = config.WHISPER_REQUEST_TIMEOUT,
    ):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def health(self, timeout: float = 2.0) -> Optional[dict]:
        """The child's readiness, or None if it is not answering yet."""
        try:
            r = httpx.get(f"{self.base}/health", timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    async def transcribe(
        self,
        path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        word_timestamps: bool = False,
        service_owned: bool = False,
    ) -> dict:
        body = {
            "path": path,
            "language": language,
            "prompt": prompt,
            "temperature": temperature,
            "word_timestamps": word_timestamps,
        }
        try:
            timeout = (httpx.Timeout(None,connect=10,write=10,pool=10)
                       if service_owned else self.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.base}/transcribe", json=body)
        except Exception as e:
            raise WhisperUnavailable(f"{self.base} did not answer: {e}") from e

        if r.status_code == 200:
            return r.json()

        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise WhisperFailed(str(detail), r.status_code)
