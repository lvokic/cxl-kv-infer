"""Current NUMA topology and affinity for individual CPU workers; no allocation."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re


def parse_list(text):
    result = set()
    for part in text.strip().split(','):
        if not part:
            continue
        bounds = [int(v) for v in part.split('-')]
        if len(bounds) > 2 or bounds[-1] < bounds[0]:
            raise ValueError(f'invalid kernel range: {part}')
        result.update(range(bounds[0], bounds[-1] + 1))
    return tuple(sorted(result))


@dataclass(frozen=True)
class NumaNode:
    node: int
    cpus: tuple[int, ...]
    memory_bytes: int


class Topology:
    def __init__(self):
        root = Path('/sys/devices/system/node')
        self.online = parse_list((root / 'online').read_text())
        self.nodes = {}
        for node in self.online:
            directory = root / f'node{node}'
            match = re.search(r'MemTotal:\s+(\d+)\s+kB', (directory / 'meminfo').read_text())
            self.nodes[node] = NumaNode(node, parse_list((directory / 'cpulist').read_text()),
                                        int(match[1]) * 1024 if match else 0)
        self.allowed_cpus = tuple(sorted(os.sched_getaffinity(0)))

    def require_nodes(self, requested, *, memory_only=False):
        if not requested or len(set(requested)) != len(requested):
            raise ValueError('request explicit, unique NUMA nodes')
        missing = set(requested) - set(self.nodes)
        if missing:
            raise RuntimeError(f'requested NUMA nodes are offline: {sorted(missing)}; no replacement or DRAM fallback')
        for node in requested:
            if self.nodes[node].memory_bytes == 0:
                raise RuntimeError(f'node {node} has no observed memory')
            if memory_only and self.nodes[node].cpus:
                raise ValueError(f'configured CXL node {node} has CPUs; confirm the platform mapping')

    def require_workers(self, cpus, cpu_nodes, *, allow_smt=False):
        self.require_nodes(cpu_nodes)
        available = {cpu for node in cpu_nodes for cpu in self.nodes[node].cpus} & set(self.allowed_cpus)
        if not cpus or len(set(cpus)) != len(cpus) or not set(cpus) <= available:
            raise ValueError('worker CPUs must be unique allowed CPUs from the configured CPU/DRAM tier')
        if not allow_smt:
            used = set()
            for cpu in cpus:
                path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
                siblings = parse_list(path.read_text())
                if used.intersection(siblings):
                    raise ValueError('multiple SMT siblings requested; enable allow_smt explicitly')
                used.update(siblings)

    def snapshot(self):
        return {'online_nodes': list(self.online), 'allowed_cpus': list(self.allowed_cpus),
                'nodes': {str(n): asdict(v) for n, v in self.nodes.items()}}


@contextmanager
def worker_affinity(cpus, cpu_nodes=(0, 1), *, allow_smt=False):
    """Bind the calling Linux thread, restoring its prior affinity on exit."""
    topology = Topology()
    topology.require_workers(cpus, cpu_nodes, allow_smt=allow_smt)
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, set(cpus))
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)
