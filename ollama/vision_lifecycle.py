"""Stop only this manager's drained vision child; retain claims on uncertainty."""
import subprocess
import logging
import time


def stop_owned_vision(model):
    """A cleanup observation bound never authorizes release by elapsed time.

    The caller owns model admission and has drained work. Popen identity, child
    exit/reap, full backend inventory and refused port must all agree. A live
    or unobservable child leaves the model/claim linked for reconciliation.
    """
    from .config import VLM_HOST, VLM_PORT
    from .inference_shared_resource import cold_vision_absent
    process = model.vlm_process
    args = getattr(process, 'args', None)
    expected = ['-m', 'mlx_vlm.server', '--model', model.name,
                '--port', str(VLM_PORT), '--host', VLM_HOST]
    if (not model.managed or process is None or type(process.pid) is not int
            or process.pid <= 0 or not isinstance(args, (list, tuple))
            or list(args[1:]) != expected or model.port != VLM_PORT):
        raise RuntimeError('Vision process ownership is unconfirmed')
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
        if process.poll() is None or cold_vision_absent() is not True:
            raise RuntimeError('Vision backend absence is unconfirmed')
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError('Vision termination requires reconciliation') from None


def recover_owned_vision(model, *, confirm_admission, resume=False):
    """Replace a proven ended child without giving up its retained claim.

    The service has drained native work. Its callback freshly verifies this
    exact model/lease against the registry before spawn and before restoring
    readiness. Existing renewal covers startup; health probes do not poll the
    registry each second. Neither failure nor elapsed time releases the claim.
    The replacement handle is recorded immediately, including a failed startup.
    """
    import httpx
    from .config import VLM_HOST, VLM_PORT, VLM_PYTHON
    process = model.vlm_process
    expected = [VLM_PYTHON, '-m', 'mlx_vlm.server', '--model', model.name,
                '--port', str(VLM_PORT), '--host', VLM_HOST]
    if (model.in_flight or not model.managed or process is None
            or type(process.pid) is not int or process.pid <= 0
            or not isinstance(process.args, (list, tuple))
            or list(process.args[1:]) != expected[1:] or model.port != VLM_PORT):
        raise RuntimeError('Vision recovery requires drained work and an exact owned child')
    if process.poll() is None:
        if resume is not True:
            raise RuntimeError('Live vision child has no saved recovery intent')
        confirm_admission()
    else:
        stop_owned_vision(model)
        confirm_admission()
        process = subprocess.Popen(expected, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        model.vlm_process = process
    started = time.monotonic()
    next_report = started
    while True:
        if process.poll() is not None:
            raise RuntimeError('Replacement vision child exited; retained claim requires reconciliation')
        try:
            ready = httpx.get(f'http://{VLM_HOST}:{VLM_PORT}/health', timeout=2).status_code == 200
        except httpx.HTTPError:
            ready = False
        if ready:
            confirm_admission()
            return
        observed = time.monotonic()
        if observed >= next_report:
            logging.getLogger(__name__).info(
                'Vision recovery: pid %s alive, health not ready, %.0fs elapsed',
                process.pid, observed - started)
            next_report = observed + 30
        time.sleep(1)
