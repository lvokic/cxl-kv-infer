"""Attention execution backends for shared group and full-layer requests."""

import numpy as np
import torch
import torch.nn.functional as F

from reference import selected_attention
from .interfaces import AttentionRequest, AttentionResult


class SDPAAttentionExecutor:
    name = "sdpa"

    def execute(self, request: AttentionRequest) -> AttentionResult:
        if request.rows is None:
            from transformers.integrations.sdpa_attention import sdpa_attention_forward

            output, _ = sdpa_attention_forward(
                request.module,
                request.query,
                request.keys,
                request.values,
                request.attention_mask,
                scaling=request.scale,
                dropout=request.dropout,
                **(request.backend_kwargs or {}),
            )
            return AttentionResult(output)
        query, keys, values = request.query, request.keys, request.values
        mask = torch.zeros(
            (query.shape[0], query.shape[1], query.shape[2], keys.shape[2]),
            dtype=torch.bool,
            device=query.device,
        )
        for head, rows in enumerate(request.rows):
            mask[:, head, :, rows] = True
        output = F.scaled_dot_product_attention(
            query,
            keys.expand(-1, query.shape[1], -1, -1),
            values.expand(-1, query.shape[1], -1, -1),
            attn_mask=mask,
            is_causal=False,
            dropout_p=request.dropout,
            scale=request.scale,
        )
        return AttentionResult(output.transpose(1, 2).contiguous())


class FlashInferAttentionExecutor:
    name = "flashinfer"

    def __init__(self, workspace_bytes: int = 128 * 1024 * 1024):
        self.workspace_bytes = workspace_bytes
        self.workspace = None
        self.wrapper = None

    def execute(self, request: AttentionRequest) -> AttentionResult:
        query, keys, values = request.query, request.keys, request.values
        if not query.is_cuda or query.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashInfer decode requires CUDA BF16/FP16 tensors")
        if query.shape[-1] not in (64, 128):
            raise ValueError("FlashInfer integration supports head_dim 64/128 only")
        if request.rows is None:
            raise ValueError("FlashInfer executor currently requires per-head selected rows")
        if self.wrapper is None:
            from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

            self.workspace = torch.zeros(
                self.workspace_bytes, dtype=torch.uint8, device=query.device
            )
            self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, kv_layout="NHD", use_tensor_cores=False, backend="fa2"
            )
        union = torch.unique(torch.cat(list(request.rows)), sorted=True)
        indices = torch.cat([torch.searchsorted(union, rows) for rows in request.rows]).to(
            torch.int32
        )
        sizes = [rows.numel() for rows in request.rows]
        indptr = torch.tensor(
            [0, *np.cumsum(sizes).tolist()], dtype=torch.int32, device=query.device
        )
        last = torch.ones(query.shape[1], dtype=torch.int32, device=query.device)
        self.wrapper.plan(
            indptr,
            indices,
            last,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=query.shape[-1],
            page_size=1,
            pos_encoding_mode="NONE",
            sm_scale=request.scale,
            q_data_type=query.dtype,
            kv_data_type=keys.dtype,
        )
        output = self.wrapper.run(
            query[0, :, 0].contiguous()[:, None, :],
            (
                keys[0, 0, union].contiguous()[:, None, None, :],
                values[0, 0, union].contiguous()[:, None, None, :],
            ),
        )
        return AttentionResult(output[:, 0][None, None, :, :])


class ReferenceAttentionExecutor:
    name = "reference"

    def execute(self, request: AttentionRequest) -> AttentionResult:
        if request.trace is None:
            raise ValueError("FP64 reference execution requires a GroupTrace")
        state = selected_attention(request.trace)
        output = torch.from_numpy(np.asarray(state.output)).to(
            device=request.query.device, dtype=request.query.dtype
        )[None, None]
        return AttentionResult(output)


def default_executors() -> dict[str, object]:
    return {
        "sdpa": SDPAAttentionExecutor(),
        "flashinfer": FlashInferAttentionExecutor(),
        "reference": ReferenceAttentionExecutor(),
    }
