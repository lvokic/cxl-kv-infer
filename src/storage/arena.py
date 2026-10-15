"""Budgeted, single-node extents for separate K/V owners. Native import is lazy."""

import importlib
import mmap
from threading import RLock

from infra.topology import Topology


class TierAllocator:
    """One extent belongs to one node. Round-robin does not interleave a process.

    The returned native owner and every NumPy view keep the byte reservation alive.
    Drop all views and owners to reclaim an extent. This is synchronous host memory;
    GPU/DMA users must retain the owner until their real completion event fires.
    """

    def __init__(self, ledger, *, cpu_nodes=(0, 1), cxl_nodes=(2, 3)):
        self.ledger = ledger
        self.nodes = {'cpu_dram': tuple(cpu_nodes), 'cxl': tuple(cxl_nodes)}
        self._cursor = {}
        self._lock = RLock()

    def allocate(self, tier, size, *, purpose, nodes=None):
        if tier not in self.nodes or type(size) is not int or size <= 0:
            raise ValueError('provide a logical host tier and positive integer size')
        nodes = self.nodes[tier] if nodes is None else tuple(nodes)
        if not set(nodes) <= set(self.nodes[tier]):
            raise ValueError('requested nodes are outside the selected tier')
        Topology().require_nodes(nodes, memory_only=tier == 'cxl')
        try:
            native = importlib.import_module('storage._numa')
        except ImportError as exc:
            raise RuntimeError('build the NUMA bridge with scripts/build_native.py; no ordinary-host fallback') from exc
        mapped = ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        with self._lock:
            cursor = self._cursor.get((tier, nodes), 0)
            node = nodes[cursor % len(nodes)]
            self._cursor[tier, nodes] = cursor + 1
            charge = self.ledger.reserve(purpose, {f'node:{node}': mapped})
        try:
            region = native.Region(size, node)
            region._reservation = charge
            region.tier = tier
            region.purpose = purpose
            return region
        except BaseException:
            charge.release()
            raise

    def allocate_kv(self, size_each, *, space, tier, nodes=None):
        """Independent extents; no aliasing K and V ownership or their budgets."""
        key = self.allocate(tier, size_each, purpose=f'{space}:canonical_K', nodes=nodes)
        value = self.allocate(tier, size_each, purpose=f'{space}:canonical_V', nodes=nodes)
        return key, value
