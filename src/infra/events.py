"""Append-only causal session traces and CPU stage records; no model imports."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time
from threading import get_ident

import numpy as np


class SessionJournal:
    """Save each KV append once, with independent K/V files and ordered Q events.

    This single-writer journal records caller-provided real or synthetic events. It does not create a
    persistent model session, enforce GPU readiness, or count physical CXL traffic.
    """

    def __init__(self, directory, metadata):
        required = {'origin', 'model_id', 'model_revision', 'position_transform', 'source_dtype'}
        if not required <= set(metadata) or metadata['origin'] not in ('model', 'synthetic', 'repeated_trace'):
            raise ValueError('record explicit input origin and model/position/precision provenance')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._stream = (self.directory / 'events.jsonl').open('x', encoding='utf-8')
        self._sequence = 0
        self._visible = {}
        self._layouts = {}
        self._writer = get_ident()
        self._write('start', metadata=dict(metadata), clock='perf_counter_ns; CPU monotonic domain')

    def _write(self, kind, **fields):
        if get_ident() != self._writer:
            raise RuntimeError('session journal requires one writer; queue worker events to its owner')
        if {'sequence', 'kind', 'cpu_monotonic_ns'} & fields.keys():
            raise ValueError('event fields cannot overwrite sequence, type or clock identity')
        record = {'sequence': self._sequence, 'kind': kind, 'cpu_monotonic_ns': time.perf_counter_ns(), **fields}
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self._stream.flush()
        self._sequence += 1

    def _array(self, name, values):
        if get_ident() != self._writer:
            raise RuntimeError('array capture requires the journal writer')
        path = self.directory / name
        with path.open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
        return path.name

    def append(self, space, keys, values, positions):
        if (not isinstance(keys, np.ndarray) or not isinstance(values, np.ndarray) or
                keys.shape != values.shape or keys.ndim != 2 or not len(keys) or
                keys.dtype.kind != 'f' or values.dtype != keys.dtype or
                not np.isfinite(keys).all() or not np.isfinite(values).all()):
            raise ValueError('append expects finite matching [tokens, dimension] floating K/V')
        first = self._visible.get(space, 0)
        layout = (keys.shape[1], keys.dtype)
        if space in self._layouts and self._layouts[space] != layout:
            raise ValueError('KV dimension/precision changed within the same space identity')
        if (not isinstance(positions, np.ndarray) or positions.dtype.kind not in 'iu' or
                not np.array_equal(positions, np.arange(first, first + len(keys), dtype=np.int64))):
            raise ValueError('append must extend contiguous positions without holes or duplicate tokens')
        stem = f'{self._sequence:08d}'
        paths = {'keys': self._array(stem + '.k.npy', keys),
                 'values': self._array(stem + '.v.npy', values),
                 'positions': self._array(stem + '.positions.npy', positions)}
        self._write('append', space=asdict(space), first_id=first, visible_hi=first + len(keys),
                    paths=paths, dtype=str(keys.dtype), logical_payload_bytes=keys.nbytes + values.nbytes,
                    source_placement='caller supplied; placement requires separate evidence')
        self._visible[space] = first + len(keys)
        self._layouts[space] = layout

    def query(self, space, queries, *, position, turn, step, scale):
        visible = self._visible.get(space, 0)
        if (not 0 <= position < visible or not isinstance(queries, np.ndarray) or
                queries.ndim != 2 or not len(queries) or queries.dtype.kind != 'f' or not np.isfinite(queries).all() or
                not np.isfinite(scale) or scale <= 0 or min(turn, step) < 0):
            raise ValueError('query must be finite, identity-tagged, and inside the already appended prefix')
        if (queries.shape[1], queries.dtype) != self._layouts[space]:
            raise ValueError('Q dimension/precision does not match the appended KV space')
        path = self._array(f'{self._sequence:08d}.q.npy', queries)
        self._write('query', space=asdict(space), queries=path, query_position=position,
                    visible_hi=position + 1, available_hi=visible, turn=turn, step=step, scale=float(scale))

    def lifecycle(self, space, action, *, generation, **fields):
        if action not in ('build', 'catch_up', 'publish', 'retire', 'reclaim', 'backpressure') or generation < 0:
            raise ValueError('record an explicit lifecycle action and generation')
        self._write('lifecycle', space=asdict(space), action=action, generation=generation, **fields)

    @contextmanager
    def stage(self, name, **fields):
        started = time.perf_counter_ns()
        try:
            yield
        except BaseException:
            self._write('stage', stage=name, status='failed', wall_ns=time.perf_counter_ns() - started, **fields)
            raise
        else:
            self._write('stage', stage=name, status='completed', wall_ns=time.perf_counter_ns() - started, **fields)

    def close(self):
        if not self._stream.closed:
            self._write('end', visible=[{'space': asdict(k), 'visible_hi': v} for k, v in self._visible.items()])
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
