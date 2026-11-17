"""Drained, manager-owned Whisper cleanup; unknown exit retains its claim."""
import math
import subprocess


def cold_whisper_absent():
    """Require full process inventory and refused backend connection together."""
    from .config import WHISPER_HOST, WHISPER_PORT
    from .inference_bootstrap import vision_backend_absent
    try:
        result = subprocess.run(['/bin/ps', '-axo', 'command='], capture_output=True,
                                text=True, timeout=5)
        if result.returncode or not result.stdout.strip():
            return False
        for line in result.stdout.splitlines():
            if any(part.rsplit('/', 1)[-1] == 'whisper_server.py' for part in line.split()):
                return False
        return vision_backend_absent(WHISPER_HOST, WHISPER_PORT) is True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def stop_owned_whisper(model, *, observation_seconds):
    """Signal one owned idle child; release requires positive exit and absence.

    The caller explicitly supplies a cleanup observation bound. Expiry neither
    kills the child nor releases bytes; it leaves the model linked for the
    service's existing reconciliation. There is no compute deadline or retry.
    """
    from . import config
    from .manager import WHISPER_PYTHON, WHISPER_SCRIPT, match_whisper_model
    match = match_whisper_model(model.name)
    process = model.whisper_process
    args = getattr(process, 'args', None)
    if (isinstance(observation_seconds, bool) or not isinstance(observation_seconds, (int, float))
            or not math.isfinite(observation_seconds) or observation_seconds <= 0):
        raise ValueError('explicit positive cleanup observation bound required')
    expected = None if match is None else [WHISPER_PYTHON, WHISPER_SCRIPT,
        '--model', match[1], '--host', config.WHISPER_HOST,
        '--port', str(config.WHISPER_PORT), '--spool', str(config.WHISPER_SPOOL_DIR),
        '--ffmpeg', config.FFMPEG_BIN]
    if (not model.managed or model.in_flight != 0 or process is None
            or type(process.pid) is not int or process.pid <= 0
            or not isinstance(args, (list, tuple)) or list(args) != expected
            or model.port != config.WHISPER_PORT):
        raise RuntimeError('Whisper process ownership or drained work is unconfirmed')
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=observation_seconds)
        if process.poll() is None or cold_whisper_absent() is not True:
            raise RuntimeError('Whisper backend absence is unconfirmed')
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError('Whisper termination requires reconciliation') from None
