"""Independent KV row selection policies."""

from collections.abc import Sequence

import torch

from .interfaces import KVCacheLease


class FullSelector:
    name = "full"

    def select(self, query, lease: KVCacheLease, scale: float) -> Sequence[torch.Tensor]:
        count = lease.keys.shape[-2]
        return [torch.arange(count, device=query.device) for _ in range(query.shape[1])]


class ExactTopKSelector:
    """Dense FP32 per-head ranking with mandatory sink/window/current rows."""

    name = "exact-topk"

    def __init__(self, top_k: int, sink_tokens: int, window_tokens: int):
        self.top_k = top_k
        self.sink_tokens = sink_tokens
        self.window_tokens = window_tokens

    def select(self, query, lease: KVCacheLease, scale: float) -> Sequence[torch.Tensor]:
        token_count = lease.keys.shape[-2]
        mandatory = torch.zeros(token_count, dtype=torch.bool, device=query.device)
        mandatory[: self.sink_tokens] = True
        if self.window_tokens:
            mandatory[max(0, token_count - self.window_tokens) :] = True
        mandatory[-1] = True
        required = torch.where(mandatory)[0]
        history = torch.where(~mandatory)[0]
        count = min(self.top_k, history.numel())
        scores = (query[0, :, 0].float() @ lease.keys[0, 0].float().T) * scale
        rank = torch.argsort(scores[:, history], dim=-1, descending=True, stable=True)
        return [
            torch.cat((required, history[rank[head, :count]])).sort().values
            for head in range(query.shape[1])
        ]


class QuestSelector:
    """Quest-style per-head chunk upper-bound selection.

    The score bound and minimum-three-chunks rule follow MInference's Quest
    decode implementation. The selected rows are then passed to this project's
    executor, which recomputes exact attention logits from the leased K rows.
    """

    name = "quest"

    def __init__(self, token_budget: int, chunk_size: int):
        if token_budget < 0 or chunk_size < 1:
            raise ValueError("Quest token budget must be nonnegative and chunk size positive")
        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.last_metrics = {}

    def select(self, query, lease: KVCacheLease, scale: float) -> Sequence[torch.Tensor]:
        keys = lease.keys[0, 0].float()
        q = query[0, :, 0].float()
        count, dim = keys.shape
        blocks = (count + self.chunk_size - 1) // self.chunk_size
        pad = blocks * self.chunk_size - count
        if pad:
            keys = torch.cat(
                (keys, torch.zeros((pad, dim), device=keys.device)), dim=0
            )
        # For each query, max(K * sign(Q)) yields an upper bound on Q.K per chunk.
        signs = torch.where(q >= 0, 1.0, -1.0)
        positive_q = q * signs
        signed_keys = keys[None, :, :] * signs[:, None, :]
        if pad:
            signed_keys[:, count:, :] = -torch.inf
        chunk_max = signed_keys.reshape(
            q.shape[0], blocks, self.chunk_size, dim
        ).amax(dim=2)
        scores = torch.einsum("hd,hbd->hb", positive_q, chunk_max) * scale
        take_blocks = min(max(3, self.token_budget // self.chunk_size), blocks)
        selected = []
        for head in range(q.shape[0]):
            chosen_blocks = torch.argsort(
                scores[head], descending=True, stable=True
            )[:take_blocks]
            rows = (
                chosen_blocks[:, None] * self.chunk_size
                + torch.arange(self.chunk_size, device=q.device)[None, :]
            ).reshape(-1)
            rows = rows[rows < count].sort().values
            selected.append(rows)
        self.last_metrics = {
            "algorithm": "quest_chunk_upper_bound",
            "chunk_size": self.chunk_size,
            "token_budget": self.token_budget,
            "candidate_rows": count,
            "selected_per_head": [int(rows.numel()) for rows in selected],
            "ranking_dtype": "float32",
            "chunk_summary_rebuilt_each_call": True,
        }
        return selected


class RetrievalAttentionSelector:
    """MInference RetrievalAttention's exact Flat-IP candidate stage.

    MInference retains sink/recent tokens on GPU and searches historical K with
    a CPU FlatIndex. This adapter reproduces that candidate policy using an
    exhaustive NumPy inner-product search. The normal executor recomputes exact
    QK logits for selected rows instead of treating index distances as logits.
    """

    name = "retr-attn"

    def __init__(self, top_k: int, sink_tokens: int, recent_tokens: int):
        if min(top_k, sink_tokens, recent_tokens) < 0:
            raise ValueError("retrieval budgets must be nonnegative")
        self.top_k = top_k
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.last_metrics = {}

    def select(self, query, lease: KVCacheLease, scale: float) -> Sequence[torch.Tensor]:
        import numpy as np

        keys = lease.keys[0, 0]
        q = query[0, :, 0]
        count = keys.shape[0]
        sink_end = min(self.sink_tokens, count)
        recent_start = max(sink_end, count - self.recent_tokens)
        mandatory = torch.cat(
            (
                torch.arange(sink_end, device=q.device),
                torch.arange(recent_start, count, device=q.device),
            )
        ).unique(sorted=True)
        history = torch.arange(sink_end, recent_start, device=q.device)
        take = min(self.top_k, history.numel())
        if take:
            # This explicit host copy models MInference's CPU FlatIndex placement.
            # Its bytes/time are exposed in selector metrics, not hidden as free work.
            host_keys = keys[history].detach().float().cpu().numpy()
            host_query = q.detach().float().cpu().numpy()
            scores = host_query @ host_keys.T
            candidate_order = np.argsort(-scores, axis=-1, kind="stable")[:, :take]
            history_host = history.detach().cpu().numpy()
            result = [
                torch.as_tensor(
                    np.sort(history_host[candidate_order[head]]),
                    dtype=torch.long,
                    device=q.device,
                )
                for head in range(q.shape[0])
            ]
        else:
            host_query = None
            host_keys = None
            result = [
                torch.empty(0, dtype=torch.long, device=q.device)
                for _ in range(q.shape[0])
            ]
        selected = [torch.cat((mandatory, rows)).unique(sorted=True) for rows in result]
        self.last_metrics = {
            "algorithm": "minference_retr_attn_flat_ip_candidates",
            "index": "exhaustive_cpu_inner_product",
            "top_k": self.top_k,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
            "candidate_rows": int(history.numel()),
            "selected_per_head": [int(rows.numel()) for rows in selected],
            "host_key_copy_bytes": 0 if host_keys is None else int(host_keys.nbytes),
            "host_query_copy_bytes": 0 if host_query is None else int(host_query.nbytes),
            "search_score_scale_applied": False,
        }
        return selected


def build_selector(
    name: str,
    *,
    top_k: int,
    sink_tokens: int,
    window_tokens: int,
    chunk_size: int = 16,
    retrieval_sink_tokens: int = 128,
    retrieval_recent_tokens: int = 512,
):
    if name == "full":
        return FullSelector()
    if name == "exact-topk":
        return ExactTopKSelector(top_k, sink_tokens, window_tokens)
    if name == "quest":
        return QuestSelector(top_k, chunk_size)
    if name in ("retr-attn", "retr_attn"):
        return RetrievalAttentionSelector(
            top_k, retrieval_sink_tokens, retrieval_recent_tokens
        )
    raise ValueError(f"unknown selector: {name}")
