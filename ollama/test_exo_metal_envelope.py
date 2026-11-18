"""CPU metadata controls; no Metal import, allocation or registry operation."""
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import capacity
from .exo_groups import load_capacity


class MetalPolicyControls(unittest.TestCase):
    def reading(self, **changes):
        return {'memory_total_mb': 36864, 'memory_device_inuse_mb': 600,
                'memory_available_mb': 100, **changes}

    def query(self, limits):
        def query(name, result, size, _new, _newsize):
            ctypes.cast(result, ctypes.POINTER(ctypes.c_uint64))[0] = limits[name.decode()]
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 8
            return 0
        return query

    def observe(self, reading, *, limit=30720, pages=250000):
        library = SimpleNamespace(sysctlbyname=self.query(
            {'iogpu.wired_limit_mb': limit, 'hw.memsize': 36864 * 1024 * 1024}))
        with patch.object(load_capacity.sys, 'platform', 'darwin'), \
                patch.object(load_capacity.ctypes, 'CDLL', return_value=library), \
                patch('ollama.exo_groups.staging_files.darwin_boot_id', return_value='fixture-boot'), \
                patch.object(capacity, '_vm_stat', return_value=({'wired down': pages}, 16384)):
            return load_capacity.metal_policy(reading)

    def test_effective_wire_policy_does_not_depend_on_instant_free_ram(self):
        first = self.observe(self.reading(memory_available_mb=0))
        second = self.observe(self.reading(memory_available_mb=20000))
        self.assertEqual(first, second)
        self.assertEqual(first, {'boot_id': 'fixture-boot', 'wired_limit_bytes': 30720 * 1024 * 1024,
            'kernel_wired_bytes': 250000 * 16384, 'device_inuse_bytes': 600 * 1024 * 1024})

    def test_automatic_unconfirmed_limit_cannot_silently_become_a_fixed_envelope(self):
        for limit in (0, 40000):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.observe(self.reading(), limit=limit)

    def test_missing_or_foreign_device_counters_refuse(self):
        for changes in ({'memory_device_inuse_mb': None}, {'memory_device_inuse_mb': True},
                        {'memory_device_inuse_mb': -1}, {'memory_total_mb': 20000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.observe(self.reading(**changes))

    def test_failed_kernel_read_is_unknown_not_a_zero_value_or_cached_success(self):
        def failed(*_args):
            return -1
        library = SimpleNamespace(sysctlbyname=failed)
        with patch.object(load_capacity.sys, 'platform', 'darwin'), \
                patch.object(load_capacity.ctypes, 'CDLL', return_value=library), self.assertRaises(ValueError):
            load_capacity.metal_policy(self.reading())
