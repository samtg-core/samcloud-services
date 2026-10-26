"""Native ContextEngine selection; never auto-activated by the general loader."""
from ollama.hermes_queue_context import QueuedContextCompressor


def register(ctx):
    ctx.register_context_engine(QueuedContextCompressor())
