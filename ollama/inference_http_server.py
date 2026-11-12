"""Close transport observers before the normal owned-compute shutdown drain."""
from starlette.responses import JSONResponse
import uvicorn

from .inference_supervisor import ObserverClosed


async def observer_closed_response(request, error: ObserverClosed):
    return JSONResponse(status_code=503, content={'detail': {
        'code': 'service_restarting',
        'execution_id': error.execution_id,
        'message': 'Resume observation of the saved execution'}},
        headers={'X-Inference-Execution': error.execution_id})


class QueueServer(uvicorn.Server):
    def __init__(self, config, *, get_runtime):
        super().__init__(config)
        self.get_runtime = get_runtime

    async def shutdown(self, sockets=None):
        runtime = self.get_runtime()
        if runtime is not None:
            runtime.begin_http_shutdown()
        # Uvicorn closes listeners/connections, then invokes ASGI lifespan.
        # Lifespan owns compute draining and exact release; this hook does not.
        await super().shutdown(sockets=sockets)
