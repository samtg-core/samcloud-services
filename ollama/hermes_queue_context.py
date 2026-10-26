"""Stock compressor with an explicitly owned queued-summary scope (opt-in).

No summary algorithm, session commit, threshold derivation or stop-fence rewrite.
The current Qwen profile's ordinary compression configuration is projected onto
the stock constructor; constructor-only host overrides require separate parity
qualification before activation. No live profile is changed by this module.
"""
from copy import deepcopy
import logging

from agent.context_compressor import ContextCompressor

from .hermes_queue_provider import MAIN_PROFILES, host_for_home


def configured_options():
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly()
    compression = cfg.get('compression', {})
    model = cfg.get('model', {})
    model = model if isinstance(model, dict) else {'default': model}
    rename = {'threshold': 'threshold_percent', 'target_ratio': 'summary_target_ratio',
              'threshold_tokens': 'threshold_tokens_cap'}
    same = ('protect_first_n', 'protect_last_n', 'abort_on_summary_failure',
            'model_thresholds', 'proactive_prune_tokens', 'proactive_prune_min_result_chars',
            'proactive_prune_min_reclaim_tokens', 'min_tail_user_messages', 'tail_mode')
    options = {rename.get(key, key): deepcopy(compression[key])
               for key in (*rename, *same) if key in compression}
    options.update(model=model.get('default', ''), provider=model.get('provider', ''),
        base_url=model.get('base_url', ''), config_context_length=model.get('context_length'),
        custom_providers=deepcopy(cfg.get('custom_providers')))
    return options


class QueuedContextCompressor(ContextCompressor):
    def __init__(self, **options):
        from hermes_constants import get_hermes_home
        self._queue_home = get_hermes_home().resolve()
        options = options or configured_options()
        self._queue_constructor_options = deepcopy(options)
        super().__init__(**options)

    @property
    def name(self):
        return 'samcloud-queue-context'

    def clone_for_agent(self):
        # Never deepcopy a bound native SessionDB or any mutable session state.
        return type(self)(**deepcopy(self._queue_constructor_options))

    def compress(self, messages, current_tokens=None, focus_topic=None, **kwargs):
        host = host_for_home(self._queue_home)
        # Relay is bound by the native turn facade, not an observer hook worker.
        # Revalidate it at submission so copied contexts cannot revive a closed turn.
        owner_provider = None
        if self.provider == 'samcloud-queue-stint' or (host.job_continuity and self.provider in MAIN_PROFILES):
            from agent.relay_runtime import active_turn
            turn = active_turn(self._session_id)
            if turn is not None and turn.lease.profile_key == host.scope:
                turn_id, session_id, provider = turn.turn_id, self._session_id, self.provider
                def owner_provider():
                    current = active_turn(session_id)
                    if (current is turn and not current.closed and not current.lease.released
                            and self._session_id == session_id and self.provider == provider):
                        return turn_id, self.model
                    return None
        with host.compression_scope(self._session_id, messages, persist_unstored=True,
                                    owner_provider=owner_provider) as source:
            self.last_queue_source_id = getattr(source, 'operation', None)
            if self.last_queue_source_id:
                logging.getLogger(__name__).info('Queued compaction source: %s', self.last_queue_source_id)
            return super().compress(messages, current_tokens=current_tokens,
                focus_topic=focus_topic, **kwargs)
