"""A retired/refused allocation can change size; unfinished ownership cannot."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from .inference_arbiter import CapacityDeferred, Conflict
from .inference_reservation import InitialReservations


class ReservationRedimension(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'initial.sqlite'
        self.receipt = dict(status_code=200, status='active', lease_id='old',
                            expires_at='2099-01-01T00:00:00Z')
        self.client = SimpleNamespace(request_lease=Mock(return_value=self.receipt),
            list_leases=Mock(return_value=[dict(id='old', status='released',
                resource_id='gpu', service_id='service', agent_id='owner',
                memory_mb=100, exclusive=0)]), release_lease=Mock())
        self.store = self.open()

    def open(self, **changes):
        args = dict(registry_owner='owner', service_id='service', models={'m': 'gpu'})
        return InitialReservations(self.path, self.client, **(args | changes))

    def reserve(self, store=None, **changes):
        args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=60)
        return (store or self.store).reserve('m', **(args | changes))

    def retire(self):
        self.reserve()
        self.store.reconcile_absent('m', confirm_absent=lambda _: True)
        self.assertEqual(self.store.snapshot('m')['state'], 'retired')

    def assert_new_allocation(self):
        before = self.store.snapshot('m')
        reopened = self.open(budgets={'m': 200})
        def send(**kwargs):
            # The new identity/dimensions must be durable before external I/O.
            with closing(sqlite3.connect('file:' + str(self.path) + '?mode=ro',
                                        uri=True)) as db:
                db.row_factory = sqlite3.Row
                saved = dict(db.execute('SELECT * FROM initial_reservations').fetchone())
            self.assertNotEqual(saved['intent'], before['intent'])
            self.assertEqual((saved['state'], saved['memory_mb'], saved['ttl_seconds'],
                              saved['lease_id']), ('unknown', 200, 120, None))
            self.assertEqual(kwargs['memory_mb'], 200)
            self.assertEqual(kwargs['ttl_seconds'], 120)
            return self.receipt | {'lease_id': 'new'}
        self.client.request_lease.side_effect = send
        self.assertEqual(self.reserve(reopened, ttl_seconds=120), 'new')
        saved = reopened.snapshot('m')
        self.assertEqual((saved['state'], saved['lease_id']), ('handed_off', 'new'))
        self.assertEqual(self.client.request_lease.call_count, 2)
        self.client.release_lease.assert_not_called()

    def test_retired_row_accepts_new_budget_and_ttl_after_positive_reconciliation(self):
        self.retire()
        self.assert_new_allocation()

    def test_refused_row_accepts_new_budget_and_ttl_with_a_new_intent(self):
        self.client.request_lease.return_value = {'status_code': 409}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        self.assertEqual(self.store.snapshot('m')['state'], 'refused')
        self.assert_new_allocation()

    def test_unfinished_rows_keep_exact_dimensions_and_never_post_a_replacement(self):
        self.reserve()
        for state in ('unknown', 'queued', 'reserved', 'handed_off', 'retiring'):
            with self.subTest(state=state):
                with self.store._owned() as db:
                    db.execute('UPDATE initial_reservations SET state=?', (state,))
                before = self.store.snapshot('m')
                with self.assertRaises(Conflict):
                    self.reserve(self.open(budgets={'m': 200}), ttl_seconds=120)
                self.assertEqual(self.store.snapshot('m'), before)
        self.client.request_lease.assert_called_once()
        self.client.list_leases.assert_not_called()
        self.client.release_lease.assert_not_called()

    def test_terminal_state_does_not_bypass_identity_or_resource_binding(self):
        self.retire()
        before = self.store.snapshot('m')
        with self.assertRaises(PermissionError):
            self.open(registry_owner='other')
        with self.assertRaises(PermissionError):
            self.open(service_id='other')
        for changes in ({'resource': 'other'}, {'service_id': 'other'},
                        {'memory_mb': True}, {'ttl_seconds': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.reserve(**changes)
        self.assertEqual(self.store.snapshot('m'), before)
        self.client.request_lease.assert_called_once()

    def test_lost_new_reply_stays_unknown_across_reopen_without_a_second_post(self):
        self.retire()
        self.client.request_lease.side_effect = TimeoutError('synthetic lost reply')
        with self.assertRaises(Conflict):
            self.reserve(self.open(budgets={'m': 200}))
        saved = self.store.snapshot('m')
        self.assertEqual(self.client.request_lease.call_count, 2)
        self.assertEqual((saved['state'], saved['memory_mb']), ('unknown', 200))
        with self.assertRaises(Conflict):
            self.reserve(self.open(budgets={'m': 200}))
        self.assertEqual(self.store.snapshot('m'), saved)
        self.assertEqual(self.client.request_lease.call_count, 2)


if __name__ == '__main__':
    unittest.main()
