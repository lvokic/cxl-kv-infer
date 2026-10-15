"""Serialized CPU FlatIP/IVF-Flat baselines, separate from index lifecycle research."""

from contextlib import contextmanager
import importlib
from threading import RLock
import time
import weakref

import numpy as np

from .interfaces import CandidateBatch, IndexSpace


def _retain_charge_until_array_dies(charge):
    # Each owning array's finalizer retains the shared charge. It is released by
    # Reservation.__del__ only after both arrays and their views have disappeared.
    del charge


class FaissCPUIndex:
    """One append-only contiguous ID prefix, protected by a lock including search.

    Standard Faiss owns an FP32 K copy in ordinary host memory. This adapter makes
    no NUMA placement claim, and does not implement snapshots, background rebuild,
    exact tail, or attention selection. Every query head retains its own candidates.
    """

    def __init__(self, space, ledger, *, dimension, kind='ivf_flat', nlist=256,
                 nprobe=16, max_vectors=65536, threads=4, seed=42,
                 training_iterations=10, workspace_bytes=64 * 1024**2):
        self.space = space
        self.ledger = ledger
        self._lock = RLock()
        self._index = self._quantizer = self._reservation = None
        self._failed = False
        if (kind not in ('flat', 'ivf_flat') or min(dimension, nlist, nprobe, max_vectors,
                threads, training_iterations) <= 0 or nprobe > nlist or workspace_bytes < 0):
            raise ValueError('invalid CPU index configuration')
        self.dimension, self.kind = dimension, kind
        self.max_vectors, self.threads = max_vectors, threads
        self.nlist, self.nprobe = nlist, nprobe
        self.seed, self.training_iterations = seed, training_iterations
        self.trained_at_hi = 0
        self.last_operation = {}
        try:
            self.faiss = importlib.import_module('faiss')
        except ImportError as exc:
            raise RuntimeError('install requirements/ann.lock; no NumPy search fallback') from exc
        # Includes the mandatory FP32 vector copy, 64-bit IVF IDs, centroids and
        # explicit workspace/headroom. Library allocator overhead is still opaque.
        self.reservation_bytes = max_vectors * (dimension * 4 + (8 if kind == 'ivf_flat' else 0))
        self.reservation_bytes += (nlist * dimension * 4 if kind == 'ivf_flat' else 0) + workspace_bytes
        self._reservation = ledger.reserve(f'{space}:faiss_{kind}_capacity_and_workspace_estimate',
                                           {'cpu_dram': self.reservation_bytes})
        try:
            if kind == 'flat':
                self._index = self.faiss.IndexFlatIP(dimension)
            else:
                self._quantizer = self.faiss.IndexFlatIP(dimension)
                self._index = self.faiss.IndexIVFFlat(self._quantizer, dimension, nlist,
                                                     self.faiss.METRIC_INNER_PRODUCT)
                self._index.nprobe = nprobe
                self._index.cp.seed = seed
                self._index.cp.niter = training_iterations
        except BaseException:
            self.close()
            raise

    def _vectors(self, values):
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or
                values.ndim != 2 or values.shape[1] != self.dimension or
                not values.flags.c_contiguous or not np.isfinite(values).all()):
            raise ValueError('provide finite contiguous [rows, dimension] FP32 arrays; conversion copies are caller-owned')
        return values

    def _ready(self):
        if self._index is None or self._failed:
            raise RuntimeError('index is closed or poisoned by a failed mutation; build a new owner')

    @contextmanager
    def _threads(self):
        previous = self.faiss.omp_get_max_threads()
        self.faiss.omp_set_num_threads(self.threads)
        try:
            yield
        finally:
            self.faiss.omp_set_num_threads(previous)

    def train(self, keys, *, committed_hi):
        keys = self._vectors(keys)
        if len(keys) != committed_hi or committed_hi > self.max_vectors:
            raise ValueError('training input must be the declared current contiguous KV prefix')
        with self._lock, self._threads():
            self._ready()
            if self.kind == 'flat':
                return
            if self._index.is_trained or self._index.ntotal:
                raise RuntimeError('retraining requires a new index owner; do not reset the serving index')
            if committed_hi < 39 * self.nlist:
                raise ValueError('insufficient visible training prefix; keep exact coverage or explicitly reduce nlist')
            started = time.perf_counter_ns()
            try:
                self._index.train(keys)
                self.trained_at_hi = committed_hi
            except BaseException:
                self._failed = True
                raise
            self.last_operation = {'stage': 'train', 'wall_ns': time.perf_counter_ns() - started,
                                   'training_hi': committed_hi, 'input_logical_bytes': keys.nbytes}

    def append(self, keys, *, first_id):
        keys = self._vectors(keys)
        with self._lock, self._threads():
            self._ready()
            if not self._index.is_trained:
                raise RuntimeError('train before adding; unpublished KV needs external exact tail coverage')
            if first_id != self._index.ntotal or first_id + len(keys) > self.max_vectors:
                raise ValueError('append IDs must extend the contiguous prefix within its reserved capacity')
            started = time.perf_counter_ns()
            ids = None
            id_charge = None
            if self.kind == 'ivf_flat':
                # Admission/allocation failures have not mutated the index.
                id_charge = self.ledger.reserve(f'{self.space}:append_id_input', {'cpu_dram': len(keys) * 8})
                try:
                    ids = np.arange(first_id, first_id + len(keys), dtype=np.int64)
                except BaseException:
                    id_charge.release()
                    raise
            try:
                if self.kind == 'flat':
                    self._index.add(keys)
                else:
                    self._index.add_with_ids(keys, ids)
            except BaseException:
                self._failed = True
                raise
            finally:
                ids = None
                if id_charge is not None:
                    id_charge.release()
            self.last_operation = {'stage': 'append', 'wall_ns': time.perf_counter_ns() - started,
                                   'indexed_hi': int(self._index.ntotal),
                                   'faiss_vector_copy_payload_bytes': keys.nbytes,
                                   'id_input_bytes': len(keys) * 8 if self.kind == 'ivf_flat' else 0}

    def search(self, queries, k, *, space, visible_hi):
        queries = self._vectors(queries)
        if space != self.space or type(k) is not int or k < 0:
            raise ValueError('query identity must match this index space; k must be nonnegative')
        with self._lock, self._threads():
            self._ready()
            if visible_hi != self._index.ntotal or visible_hi < self.trained_at_hi:
                raise ValueError('serialized baseline searches only its complete current prefix; future-trained or historical views require snapshots')
            started = time.perf_counter_ns()
            width = min(k, visible_hi)
            output_charge = self.ledger.reserve(f'{self.space}:candidate_outputs',
                                                {'cpu_dram': len(queries) * width * 12})
            try:
                if width == 0:
                    scores = np.empty((len(queries), 0), dtype=np.float32)
                    ids = np.empty((len(queries), 0), dtype=np.int64)
                else:
                    scores, ids = self._index.search(queries, width)
                weakref.finalize(scores, _retain_charge_until_array_dies, output_charge)
                weakref.finalize(ids, _retain_charge_until_array_dies, output_charge)
            except BaseException:
                output_charge.release()
                raise
            self.last_operation = {'stage': 'query', 'wall_ns': time.perf_counter_ns() - started,
                                   'query_heads': len(queries), 'visible_hi': visible_hi,
                                   'output_bytes': scores.nbytes + ids.nbytes}
            return CandidateBatch(self.space, visible_hi, ids, scores)

    def snapshot(self):
        with self._lock:
            self._ready()
            count = int(self._index.ntotal)
            return {'backend': 'faiss_cpu', 'role': 'serialized_control',
                    'kind': self.kind, 'metric': 'inner_product', 'faiss_version': self.faiss.__version__,
                    'dimension': self.dimension, 'indexed_hi': count, 'trained_at_hi': self.trained_at_hi,
                    'nlist': self.nlist if self.kind == 'ivf_flat' else None,
                    'nprobe': self.nprobe if self.kind == 'ivf_flat' else None,
                    'threads': self.threads, 'seed': self.seed, 'training_iterations': self.training_iterations,
                    'reservation_estimate_bytes': self.reservation_bytes,
                    'vector_copy_payload_estimate_bytes': count * self.dimension * 4,
                    'placement': 'ordinary host memory; Faiss allocations not NUMA verified',
                    'exact_library_reserved_bytes': None, 'concurrent_search_add': False,
                    'last_operation': dict(self.last_operation)}

    def close(self):
        with self._lock:
            self._index = self._quantizer = None
            if self._reservation is not None:
                self._reservation.release()
                self._reservation = None

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
