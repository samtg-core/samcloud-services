"""Stop only this manager's drained vision child; retain claims on uncertainty."""
import subprocess


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
