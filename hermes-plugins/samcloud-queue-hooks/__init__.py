from ollama.hermes_queue_provider import register_hooks


def register(ctx):
    register_hooks(ctx)
