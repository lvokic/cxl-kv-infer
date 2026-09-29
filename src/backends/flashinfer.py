"""A correctness adapter for independent logical heads, not a CXL benchmark.

Selected unique KV is gathered from NumPy and copied to the GPU here. This
deliberately includes allocation and blocking return; no timing is reported.
See docs/ENVIRONMENT.md for the inspected upstream version, not a tested lock.
"""

import math
import numpy as np

from reference import AttentionState, empty_state
from kv_trace import GroupTrace


def selected_attention_flashinfer(
    trace: GroupTrace, *, device: str = "cuda:0", dtype: str = "bfloat16"
) -> AttentionState:
    import torch
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    trace.validate()
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("the initial adapter supports bfloat16 or float16")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; there is no silent CPU fallback")
    result = empty_state(trace.num_heads, trace.head_dim)
    active = np.flatnonzero(np.diff(trace.indptr))
    if not active.size:
        return result
    unique_ids = np.unique(trace.indices)
    if unique_ids.size > np.iinfo(np.int32).max or trace.indices.size > np.iinfo(np.int32).max:
        raise ValueError("this adapter requires int32-compatible page metadata")
    torch_dtype = getattr(torch, dtype)

    def upload_exact(array: np.ndarray):
        cpu = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch_dtype)
        if not np.array_equal(cpu.float().numpy(), array):
            raise ValueError(
                f"conversion to {dtype} would change trace values; capture original {dtype} "
                "Q/K/V losslessly as float32, or explicitly prepare a separate converted trace"
            )
        return cpu.to(device=device)

    with torch.cuda.device(device):
        q = upload_exact(trace.q[active])[:, None, :]
        k = upload_exact(trace.k[unique_ids])[:, None, None, :]
        v = upload_exact(trace.v[unique_ids])[:, None, None, :]
        lengths = np.array([trace.selected(int(h)).size for h in active], dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        indices = np.concatenate([
            np.searchsorted(unique_ids, trace.selected(int(h))) for h in active
        ]).astype(np.int32)
        indptr_gpu = torch.as_tensor(indptr, device=device)
        indices_gpu = torch.as_tensor(indices, device=device)
        last_page_len = torch.ones(len(active), dtype=torch.int32, device=device)
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=False, backend="fa2"
        )
        wrapper.plan(
            indptr_gpu, indices_gpu, last_page_len,
            num_qo_heads=1, num_kv_heads=1, head_dim=trace.head_dim, page_size=1,
            pos_encoding_mode="NONE", sm_scale=trace.scale,
            q_data_type=torch_dtype, kv_data_type=torch_dtype,
        )
        output, lse_log2 = wrapper.run(q, (k, v), return_lse=True)
        # Canonical internal representation uses natural logarithms.
        result.output[active] = output[:, 0, :].float().cpu().numpy()
        result.lse[active] = lse_log2[:, 0].double().cpu().numpy() * math.log(2.0)
    return result
