"""Post-RoPE decode dispatch through independent selector and executor contracts."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from kv_trace import GroupTrace
from .cache import TransformersKVCacheLifecycle
from .executors import default_executors
from .interfaces import AttentionRequest
from .selectors import build_selector


class DecodeController:
    def __init__(self, config, model_info, *, cache_lifecycle=None, executors=None):
        self.config = config
        self.model_info = model_info
        self.cache_lifecycle = cache_lifecycle or TransformersKVCacheLifecycle()
        self.executors = default_executors() if executors is None else dict(executors)
        self.configure(config)
        self.phase = "prefill"
        self.workspace = None
        self.wrapper = None
        self.request_id = None
        self.prompt_hash = ""
        self.capture_dir = None
        self.capture_mode = "free_generation"
        self.step = -1
        self.query_position = -1
        self.routes, self.captures = Counter(), []
        self.selection_stats = []

    def configure(self, config):
        self.config = config
        self.selector = build_selector(
            config.selector,
            top_k=config.top_k,
            sink_tokens=config.sink_tokens,
            window_tokens=config.window_tokens,
            chunk_size=config.quest_chunk_size,
            retrieval_sink_tokens=config.retrieval_sink_tokens,
            retrieval_recent_tokens=config.retrieval_recent_tokens,
        )

    def begin_request(self, request_id, prompt_hash, capture_dir, mode="free_generation"):
        self.cache_lifecycle.begin_request(request_id)
        self.request_id, self.prompt_hash = request_id, prompt_hash
        self.capture_dir = Path(capture_dir) if capture_dir is not None else None
        self.capture_mode = mode
        self.phase, self.step, self.query_position = "prefill", -1, -1
        self.routes, self.captures = Counter(), []
        self.selection_stats = []

    def end_request(self):
        self.cache_lifecycle.end_request()

    def trace(self, q, lease, rows, layer, group, scale):
        k, v = lease.keys, lease.values
        sizes = [r.numel() for r in rows]
        token_ids = torch.cat(list(rows)).to("cpu").numpy().astype(np.int64)
        positions = lease.positions
        if positions is None:
            positions = np.arange(k.shape[2], dtype=np.int64)
        elif hasattr(positions, "detach"):
            positions = positions.detach().to("cpu").numpy().astype(np.int64)
        else:
            positions = np.asarray(positions, dtype=np.int64)
        return GroupTrace(
            q=q[0, :, 0].float().cpu().numpy(),
            k=k[0, 0].float().cpu().numpy(),
            v=v[0, 0].float().cpu().numpy(),
            indptr=np.r_[0, np.cumsum(sizes)].astype(np.int64),
            indices=token_ids,
            positions=positions,
            metadata={
                "format_version": 1,
                "origin": "model",
                "model_id": self.model_info["model_id"],
                "model_revision": self.model_info["revision"],
                "request_id": self.request_id,
                "step": self.step,
                "layer": layer,
                "kv_group": group,
                "epoch": 0,
                "query_position": self.query_position,
                "sm_scale": float(scale),
                "score_semantics": "scaled_dot_product_post_rope",
                "position_transform": self.model_info["position_transform"],
                "source_dtype": str(q.dtype).removeprefix("torch."),
                "capture_mode": self.capture_mode,
                "prompt_id": self.prompt_hash,
                "q_head_ids": list(range(group * q.shape[1], (group + 1) * q.shape[1])),
                "model_config_sha256": self.model_info["config_sha256"],
                "selector": {
                    "name": self.config.selector,
                    "historical_k": self.config.top_k,
                    "sink_tokens": self.config.sink_tokens,
                    "window_tokens": self.config.window_tokens,
                    "quest_chunk_size": self.config.quest_chunk_size,
                    "retrieval_sink_tokens": self.config.retrieval_sink_tokens,
                    "retrieval_recent_tokens": self.config.retrieval_recent_tokens,
                    "current_token_mandatory": True,
                    "ranking_dtype": "float32",
                    "tie_break": "smaller row ID",
                },
                "kv_placement": lease.placement,
                "future_decode_queries_used": False,
            },
        )

    def capture(self, q, lease, rows, output, layer, group, scale):
        if not (
            self.config.capture
            and self.capture_dir is not None
            and self.step == self.config.capture_step
        ):
            return
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        name = f"step-{self.step:04d}-layer-{layer:03d}-group-{group:03d}"
        path = self.capture_dir / (name + ".npz")
        self.trace(q, lease, rows, layer, group, scale).save(path)
        with (self.capture_dir / (name + ".attention.npy")).open("xb") as f:
            np.save(f, output[0, 0].float().cpu().numpy(), allow_pickle=False)
        self.captures.append(str(path))

    def _execute(self, name, request):
        try:
            executor = self.executors[name]
        except KeyError as exc:
            raise ValueError(f"no attention executor is registered for {name!r}") from exc
        return executor.execute(request)

    def forward(self, module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kwargs):
        layer = module.layer_idx
        target = self.config.layers is None or layer in self.config.layers
        if self.phase != "decode" or not target:
            self.routes[self.phase + "_sdpa"] += 1
            return sdpa_attention_forward(
                module, q, k, v, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )
        if q.shape[0] != 1 or q.shape[2] != 1 or k.shape[2] != self.query_position + 1:
            raise ValueError(
                "custom decode supports one unpadded request, one current query and contiguous positions"
            )
        if attention_mask is not None:
            allowed = (
                attention_mask.all()
                if attention_mask.dtype == torch.bool
                else (attention_mask == 0).all()
            )
            if not bool(allowed):
                raise ValueError(
                    "custom decode cannot ignore padding, score bias or a restricted attention mask"
                )
        scale = module.scaling if scaling is None else scaling
        n_groups, heads_per_group = k.shape[1], q.shape[1] // k.shape[1]
        full_sdpa = self.config.backend == "sdpa" and self.config.selector == "full"
        partial_layer = self.config.groups is not None and len(self.config.groups) < n_groups
        baseline = None
        if full_sdpa or partial_layer:
            # Preserve exact whole-layer rounding for controls and untargeted heads.
            result = self._execute(
                "sdpa",
                AttentionRequest(
                    query=q,
                    keys=k,
                    values=v,
                    scale=scale,
                    layer_idx=layer,
                    group_idx=None,
                    rows=None,
                    module=module,
                    attention_mask=attention_mask,
                    dropout=dropout,
                    backend_kwargs=kwargs,
                ),
            )
            baseline = result.output
            self.routes["decode_whole_layer_sdpa_control"] += 1
        pieces = []
        for group in range(n_groups):
            first, last = group * heads_per_group, (group + 1) * heads_per_group
            qg, kg, vg = q[:, first:last], k[:, group : group + 1], v[:, group : group + 1]
            selected_group = self.config.groups is None or group in self.config.groups
            if not selected_group:
                piece = baseline[:, :, first:last]
                self.routes["decode_untargeted_group_sdpa"] += 1
            else:
                lease = self.cache_lifecycle.acquire(layer, group, kg, vg, None)
                completion = None
                try:
                    rows = self.selector.select(qg, lease, scale)
                    trace = (
                        self.trace(qg, lease, rows, layer, group, scale)
                        if self.config.backend == "reference"
                        else None
                    )
                    if full_sdpa:
                        piece = baseline[:, :, first:last]
                    else:
                        result = self._execute(
                            self.config.backend,
                            AttentionRequest(
                                query=qg,
                                keys=lease.keys,
                                values=lease.values,
                                scale=scale,
                                layer_idx=layer,
                                group_idx=group,
                                rows=rows,
                                lease=lease,
                                module=module,
                                attention_mask=None,
                                dropout=dropout,
                                backend_kwargs=kwargs,
                                trace=trace,
                            ),
                        )
                        piece, completion = result.output, result.completion
                    self.routes["decode_target_" + self.config.backend] += 1
                    self.selection_stats.append(
                        {
                            "step": self.step,
                            "layer": layer,
                            "group": group,
                            "context_tokens": k.shape[2],
                            "selected_per_head": [r.numel() for r in rows],
                            "kv_logical_bytes": lease.logical_bytes,
                            "kv_transfer_bytes": lease.transfer_bytes,
                            "kv_placement": lease.placement,
                            "selector_metrics": getattr(self.selector, "last_metrics", {}),
                        }
                    )
                    self.capture(qg, lease, rows, piece, layer, group, scale)
                finally:
                    lease.release(completion)
            pieces.append(piece)
        return torch.cat(pieces, dim=2).contiguous(), None


def attention_forward(module, *args, **kwargs):
    return module._cxl_decode_controller.forward(module, *args, **kwargs)


def register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register("cxl_kv_decode", attention_forward)
    AttentionMaskInterface.register("cxl_kv_decode", sdpa_mask)
