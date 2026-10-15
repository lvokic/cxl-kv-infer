"""Explicit byte reservations. Counters are accounting, not RSS or CXL traffic."""

from threading import RLock


class BudgetExceeded(RuntimeError):
    pass


class Reservation:
    def __init__(self, ledger, identifier):
        self._ledger = ledger
        self._identifier = identifier

    def release(self):
        if self._ledger is not None:
            self._ledger._release(self._identifier)
            self._ledger = None

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class ResourceLedger:
    def __init__(self, limits, *, cpu_nodes=(0, 1), cxl_nodes=(2, 3)):
        if any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError('resource limits must be nonnegative integer bytes')
        self.limits = dict(limits)
        self._tiers = {f'node:{n}': 'cpu_dram' for n in cpu_nodes}
        self._tiers.update({f'node:{n}': 'cxl' for n in cxl_nodes})
        self._used = {k: 0 for k in limits}
        self._peak = dict(self._used)
        self._entries = {}
        self._sequence = 0
        self._lock = RLock()

    def reserve(self, purpose, charges):
        """Node charges also consume their tier budget; pinned memory charges both."""
        charges = dict(charges)
        if not purpose or not charges or any(type(v) is not int or v < 0 for v in charges.values()):
            raise ValueError('provide a purpose and nonnegative integer byte charges')
        for node, tier in self._tiers.items():
            if node in charges:
                if tier in charges:
                    raise ValueError('pass node charges or a tier charge; tier totals are computed automatically')
        totals = {}
        for node, tier in self._tiers.items():
            if node in charges:
                totals[tier] = totals.get(tier, 0) + charges[node]
        charges.update(totals)
        with self._lock:
            for key, size in charges.items():
                if key not in self.limits:
                    raise ValueError(f'no configured budget for {key}')
                if self._used[key] + size > self.limits[key]:
                    raise BudgetExceeded(f'{purpose}: {key} would use {self._used[key] + size} > {self.limits[key]} bytes')
            identifier = self._sequence
            self._sequence += 1
            self._entries[identifier] = {'purpose': purpose, 'charges': charges}
            for key, size in charges.items():
                self._used[key] += size
                self._peak[key] = max(self._peak[key], self._used[key])
            return Reservation(self, identifier)

    def _release(self, identifier):
        with self._lock:
            entry = self._entries.pop(identifier)
            for key, size in entry['charges'].items():
                self._used[key] -= size

    def snapshot(self):
        with self._lock:
            return {'scope': 'explicit reservations; excludes unregistered library/model allocations',
                    'limits_bytes': dict(self.limits), 'reserved_bytes': dict(self._used),
                    'peak_reserved_bytes': dict(self._peak),
                    'allocations': [dict(purpose=v['purpose'], charges=dict(v['charges']))
                                    for v in self._entries.values()]}
