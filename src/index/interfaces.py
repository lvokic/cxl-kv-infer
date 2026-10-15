"""Backend-independent identity and per-query-head candidate contract.

Index generations and exact tail coverage are owned by the lifecycle controller.
ANN scores rank candidates only; attention must read canonical K and recompute QK.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class IndexSpace:
    session: str
    layer: int
    group: int
    kv_epoch: int
    transform_id: str

    def __post_init__(self):
        if not self.session or not self.transform_id or min(self.layer, self.group, self.kv_epoch) < 0:
            raise ValueError('index space requires session, layer/group, KV epoch and transform identity')


@dataclass(frozen=True)
class CandidateBatch:
    space: IndexSpace
    visible_hi: int
    row_ids: np.ndarray
    scores: np.ndarray
    score_semantics: str = 'candidate inner product; recompute canonical QK before attention'


class CandidateIndex(Protocol):
    """An admitted index owner; metadata does not imply concurrent mutation safety.

    train sees only the committed prefix. append extends stable contiguous IDs.
    search returns one candidate row per Q head at the requested visible prefix;
    an adapter must reject unsupported historical views, never silently widen it.
    close requires the controller to have retired all readers first. CPU/GPU/DMA
    retirement, generation pinning and exact tail remain separate responsibilities.
    """

    space: IndexSpace

    def train(self, keys: np.ndarray, *, committed_hi: int) -> None: ...

    def append(self, keys: np.ndarray, *, first_id: int) -> None: ...

    def search(self, queries: np.ndarray, k: int, *, space: IndexSpace,
               visible_hi: int) -> CandidateBatch: ...

    def snapshot(self) -> dict: ...

    def close(self) -> None: ...
