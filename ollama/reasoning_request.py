"""Opt-in thinking for Ollama chats, shared by the direct route and the queue (task 965)."""


# Absent, or `none`, keeps the #69 default in server._chat_completions_direct:
# `think:false`, byte for byte. Exo's own forwarding of the field is untouched;
# this applies to the Ollama branch and to the queue, which serves only local
# backends. A level is passed through as Ollama's string ThinkValue: the Qwen
# renderers read it as enabled (Qwen3.8 also as its level), harmony as the level.
# These are strings api/types.go IsValid accepts in 0.33.0 and 0.40.14, less
# Ollama's own `max`.
OLLAMA_REASONING_EFFORTS = ('none', 'low', 'medium', 'high')


def reasoning_request_error(body):
    """Why a raw chat body's reasoning request must be refused, or None.

    Thinking on a queue-owned stream has no inter-token ceiling, so a level is
    only accepted with an output budget (max_tokens -> num_predict). Read from
    the RAW body: pydantic would coerce "512", 512.0 or True into a budget.
    """
    effort = body.get('reasoning_effort')
    if effort is None or effort == 'none':
        return None
    if type(effort) is not str or effort not in OLLAMA_REASONING_EFFORTS:
        return 'reasoning_effort must be one of none, low, medium, high'
    budget = body.get('max_tokens')
    if type(budget) is not int or budget <= 0:
        return 'reasoning_effort requires a positive integer max_tokens'
    return None


def ollama_think(effort):
    return False if effort is None or effort == 'none' else effort
