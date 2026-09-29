"""One request / step / layer / KV group, with immutable logical selection.

NPZ is an offline interchange format. Loading it does not place pages on CXL.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
SEMANTICS = "scaled_dot_product_post_rope"


@dataclass(frozen=True)
class GroupTrace:
    q: np.ndarray                # [query_heads, d], already position transformed
    k: np.ndarray                # [tokens, d], one shared KV head
    v: np.ndarray                # [tokens, d]
    indptr: np.ndarray           # [query_heads + 1]
    indices: np.ndarray          # final selected token rows, CSR by query head
    positions: np.ndarray        # [tokens], original positions
    metadata: dict[str, Any]

    @property
    def num_heads(self) -> int:
        return self.q.shape[0]

    @property
    def head_dim(self) -> int:
        return self.q.shape[1]

    @property
    def scale(self) -> float:
        return float(self.metadata["sm_scale"])

    def selected(self, head: int) -> np.ndarray:
        return self.indices[self.indptr[head]:self.indptr[head + 1]]

    def validate(self) -> None:
        for name in ("q", "k", "v"):
            value = getattr(self, name)
            if value.ndim != 2 or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError(f"{name} must be a 2D float32/float64 array")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.num_heads < 1 or self.head_dim < 1:
            raise ValueError("at least one query head and one dimension are required")
        if self.k.shape != self.v.shape or self.k.shape[1] != self.head_dim:
            raise ValueError("v1 requires identical K/V shapes and Q/K/V head dimensions")
        for name in ("indptr", "indices", "positions"):
            value = getattr(self, name)
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{name} must be a 1D integer array")
        if self.indptr.shape != (self.num_heads + 1,):
            raise ValueError("indptr must have num_heads + 1 entries")
        if self.indptr[0] != 0 or self.indptr[-1] != self.indices.size:
            raise ValueError("invalid CSR bounds")
        if np.any(self.indptr[1:] < self.indptr[:-1]):
            raise ValueError("indptr must be nondecreasing")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(self.k)):
            raise ValueError("selected token row is out of bounds")
        if self.positions.shape != (len(self.k),):
            raise ValueError("positions must describe every token row")
        for head in range(self.num_heads):
            ids = self.selected(head)
            if np.unique(ids).size != ids.size:
                raise ValueError(f"head {head} contains duplicate selected tokens")
        required = {
            "format_version", "origin", "model_id", "model_revision", "request_id",
            "step", "layer", "kv_group", "epoch", "query_position", "sm_scale",
            "score_semantics", "position_transform", "source_dtype",
        }
        missing = required.difference(self.metadata)
        if missing:
            raise ValueError(f"missing metadata: {sorted(missing)}")
        if self.metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported trace version")
        if self.metadata["origin"] not in ("synthetic", "model"):
            raise ValueError("origin must be synthetic or model")
        if self.metadata["score_semantics"] != SEMANTICS:
            raise ValueError("v1 supports scaled dot product on post-position-transform Q/K only")
        if self.metadata["source_dtype"] not in ("float16", "bfloat16", "float32", "float64"):
            raise ValueError("unsupported source dtype")
        if self.metadata["origin"] == "model":
            for name in ("capture_mode", "prompt_id", "q_head_ids", "model_config_sha256"):
                if name not in self.metadata:
                    raise ValueError(f"model trace needs {name}")
            heads = self.metadata["q_head_ids"]
            if len(heads) != self.num_heads or len(set(heads)) != self.num_heads:
                raise ValueError("q_head_ids must identify each query head exactly once")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("sm_scale must be finite and positive")
        if self.positions.size and np.any(self.positions < 0):
            raise ValueError("positions must be nonnegative")
        if np.unique(self.positions).size != self.positions.size:
            raise ValueError("token positions must be unique within this group snapshot")
        if self.indices.size and np.any(self.positions[self.indices] > int(self.metadata["query_position"])):
            raise ValueError("selection contains a future token")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("trace path must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally replacing captured evidence.
        with path.open("xb") as handle:
            np.savez(handle, q=self.q, k=self.k, v=self.v, indptr=self.indptr,
                     indices=self.indices, positions=self.positions,
                     metadata_json=np.asarray(json.dumps(self.metadata, allow_nan=False)))

    @classmethod
    def load(cls, path: str | Path) -> "GroupTrace":
        with np.load(path, allow_pickle=False) as data:
            trace = cls(
                **{name: data[name] for name in ("q", "k", "v", "indptr", "indices", "positions")},
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        trace.validate()
        return trace


def synthetic_trace(*, heads: int = 4, tokens: int = 128, dim: int = 128,
                    selected: int = 16, seed: int = 0) -> GroupTrace:
    """Small plumbing fixture; never a model-quality or CXL-performance workload."""
    if min(heads, tokens, dim) < 1 or not 0 <= selected <= tokens:
        raise ValueError("invalid synthetic dimensions or selected count")
    rng = np.random.default_rng(seed)
    # These binary fractions are exactly representable in FP16 and BF16.
    def values(shape: tuple[int, int]) -> np.ndarray:
        return rng.integers(-32, 33, size=shape).astype(np.float32) / 32
    trace = GroupTrace(
        q=values((heads, dim)), k=values((tokens, dim)), v=values((tokens, dim)),
        indptr=np.arange(heads + 1, dtype=np.int64) * selected,
        indices=np.concatenate([
            np.sort(rng.choice(tokens, size=selected, replace=False)) for _ in range(heads)
        ]).astype(np.int64),
        positions=np.arange(tokens, dtype=np.int64),
        metadata={
            "format_version": FORMAT_VERSION, "origin": "synthetic",
            "model_id": "synthetic", "model_revision": "none", "request_id": "synthetic",
            "step": 0, "layer": 0, "kv_group": 0, "epoch": 0,
            "query_position": tokens - 1, "sm_scale": dim ** -0.5,
            "score_semantics": SEMANTICS, "position_transform": "synthetic-no-model",
            "source_dtype": "float32", "seed": seed,
        },
    )
    trace.validate()
    return trace
