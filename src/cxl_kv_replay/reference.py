"""FP64 numerical reference, explicitly not an optimized CPU executor."""

from dataclasses import dataclass
import numpy as np

from .trace import GroupTrace


@dataclass(frozen=True)
class AttentionState:
    output: np.ndarray  # [heads, d], normalized partial output
    lse: np.ndarray     # [heads], NATURAL-log sum exp; empty partition = -inf


def empty_state(heads: int, dim: int) -> AttentionState:
    return AttentionState(np.zeros((heads, dim), dtype=np.float64),
                          np.full(heads, -np.inf, dtype=np.float64))


def selected_attention(trace: GroupTrace, membership: np.ndarray | None = None) -> AttentionState:
    """membership selects entries in the CSR indices, preserving per-head masks."""
    trace.validate()
    if membership is None:
        membership = np.ones(trace.indices.size, dtype=bool)
    if membership.dtype != np.bool_ or membership.shape != trace.indices.shape:
        raise ValueError("membership must be a bool array with indices.shape")
    state = empty_state(trace.num_heads, trace.head_dim)
    for head in range(trace.num_heads):
        lo, hi = trace.indptr[head:head + 2]
        ids = trace.indices[lo:hi][membership[lo:hi]]
        if not ids.size:
            continue
        logits = (trace.k[ids].astype(np.float64) @ trace.q[head].astype(np.float64)) * trace.scale
        maximum = logits.max()
        weights = np.exp(logits - maximum)
        normalizer = weights.sum()
        state.output[head] = (weights @ trace.v[ids].astype(np.float64)) / normalizer
        state.lse[head] = maximum + np.log(normalizer)
    return state


def merge_states(a: AttentionState, b: AttentionState) -> AttentionState:
    """Merge disjoint partitions; the caller is responsible for their provenance."""
    if a.output.shape != b.output.shape or a.lse.shape != b.lse.shape:
        raise ValueError("partial shapes differ")
    if a.output.ndim != 2 or a.lse.shape != (a.output.shape[0],):
        raise ValueError("invalid state shape")
    for state in (a, b):
        if not np.isfinite(state.output).all():
            raise ValueError("partial output must be finite")
        if not (np.isfinite(state.lse) | np.isneginf(state.lse)).all():
            raise ValueError("LSE must be finite or -inf")
    lse = np.logaddexp(a.lse, b.lse)
    output = np.zeros_like(a.output, dtype=np.float64)
    valid = np.isfinite(lse)
    # Avoid -inf - -inf for a head whose two partitions are both empty.
    output[valid] = (
        np.exp(a.lse[valid] - lse[valid])[:, None] * a.output[valid]
        + np.exp(b.lse[valid] - lse[valid])[:, None] * b.output[valid]
    )
    return AttentionState(output, lse)
