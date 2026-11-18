"""Positive host-reboot cleanup, distinct from a same-boot process exit.

This validates immutable provider evidence, never issues a lease mutation or
invented exit status. The old launch, preparation and child must already be
independently pinned. A new boot ends old process/GPU ownership; comparing a
new boot's device counter with the previous boot's baseline is not meaningful.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID

from .cleanup import CleanupExpectation, NoStartExpectation, _aware
from .http import decode_json
from .plan import BindingRefused
from .prepare import canonical


@dataclass(frozen=True)
class RebootExpectation:
    prepared: NoStartExpectation
    started: CleanupExpectation

    def __post_init__(self):
        if (not isinstance(self.prepared, NoStartExpectation)
                or not isinstance(self.started, CleanupExpectation)
                or self.prepared.binding != self.started.binding
                or self.prepared.listeners != self.started.listeners):
            raise BindingRefused('exact prepared and started member required')

    @property
    def binding(self):
        return self.started.binding


def validate_reboot_cleanup(raw, pinned, *, observed_boot_id=None):
    """No reboot proof can be accepted as an ordinary exit-code receipt."""
    if not isinstance(pinned, RebootExpectation) or not isinstance(raw, bytes) or len(raw) > 128*1024:
        raise BindingRefused('bounded reboot evidence and independent launch pin required')
    try:
        value = decode_json(raw)
        binding = asdict(pinned.binding)
        if (not isinstance(value, dict) or value.get('kind') != 'observed_rank_rebooted'
                or canonical({k:value.get(k) for k in binding}) != canonical(binding)
                or value.get('lease_released') is not False
                or value.get('settlement_recorded') is not False):
            raise ValueError('reboot identity')
        o = value['observation']
        if (not isinstance(o, dict)
                or canonical(o['binding']) != canonical(binding)
                or canonical(o['prepared']) != canonical(asdict(pinned.prepared))
                or canonical(o['child']) != canonical(asdict(pinned.started.child))
                or o['previous_host_boot'] != pinned.prepared.host_boot):
            raise ValueError('original reboot launch')
        new_boot = str(UUID(o['current_host_boot']))
        if (new_boot != o['current_host_boot'] or UUID(new_boot).int == 0
                or new_boot == pinned.prepared.host_boot
                or observed_boot_id is not None and new_boot != observed_boot_id):
            raise ValueError('fresh boot required')
        boot = o['booted_at']
        if (not isinstance(boot, dict) or set(boot) != {'seconds','microseconds'}
                or any(type(boot[k]) is not int for k in boot)
                or boot['seconds'] <= 0 or not 0 <= boot['microseconds'] < 1000000
                or (boot['seconds'],boot['microseconds']) <= pinned.started.child.birth):
            raise ValueError('boot must follow original child')
        observed = datetime.fromisoformat(_aware(o['observed_at'])).timestamp()
        if observed < boot['seconds'] + boot['microseconds']/1000000:
            raise ValueError('observation predates boot')
        ports = o['listeners_closed']
        if (type(o.get('fenced_generation')) is not int
                or o['fenced_generation'] <= pinned.binding.generation
                or o.get('kernel_inventory_complete') is not True
                or o.get('boot_rechecked') is not True
                or o.get('native_processes') != []
                or o.get('reserved_sockets') != []
                or not isinstance(ports,list) or len(ports) != len(pinned.prepared.listeners)
                or any(v is not True for v in ports)
                or any(k in o for k in ('exit_code','baseline_in_use_bytes','observed_in_use_bytes'))):
            raise ValueError('positive reboot inventory required')
        return value
    except (KeyError, TypeError, ValueError, UnicodeError, AttributeError) as error:
        raise BindingRefused('reboot cleanup differs from the exact old launch or kernel observation') from error
