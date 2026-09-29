"""KV lifecycle adapters for model-owned and future system-owned caches."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _BorrowedKVLease:
    keys: Any
    values: Any
    positions: Any
    token_ids: Any
    placement: str
    logical_bytes: int
    transfer_bytes: int
    _owner: Any = field(repr=False)
    _on_release: Any = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def release(self, completion: Any = None) -> None:
        if self._released:
            raise RuntimeError("KV lease was released more than once")
        # This adapter borrows PyTorch-owned tensors on the active stream. Other
        # cache adapters may use the completion token before returning storage.
        self._on_release(completion)
        self._released = True
        self._owner = None


class TransformersKVCacheLifecycle:
    """Observe and lease the DynamicCache that Transformers appends into.

    The cache remains the model's source of truth. This adapter never copies or
    relocates its K/V and marks its placement as ordinary CPU memory or CUDA HBM.
    """

    name = "transformers_dynamic_cache"

    def __init__(self):
        self._active = False
        self._request_id = None
        self._cache = None
        self._outstanding = 0
        self._append_calls = 0
        self._appended_tokens = 0
        self._tokens_by_layer = {}
        self._leases_issued = 0
        self._leases_released = 0

    def begin_request(self, request_id: str) -> None:
        if self._active:
            raise RuntimeError("end the active KV request before starting another")
        self._active = True
        self._request_id = request_id
        self._cache = None
        self._outstanding = 0
        self._append_calls = 0
        self._appended_tokens = 0
        self._tokens_by_layer = {}
        self._leases_issued = 0
        self._leases_released = 0

    def new_cache(self, model_config: Any) -> Any:
        if not self._active or self._cache is not None:
            raise RuntimeError("a request must have exactly one active model cache")
        from transformers import DynamicCache

        lifecycle = self

        class ObservedDynamicCache(DynamicCache):
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
                positions = (cache_kwargs or {}).get("cache_position")
                lifecycle.observe_append(
                    layer_idx, key_states, value_states, positions, (keys, values)
                )
                return keys, values

        self._cache = ObservedDynamicCache(config=model_config)
        return self._cache

    def observe_append(self, layer_idx, keys, values, positions, cache_view) -> None:
        if not self._active:
            raise RuntimeError("KV append occurred outside an active request")
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("cache append expects matching [batch, kv_heads, tokens, dim] tensors")
        if positions is not None and positions.numel() != keys.shape[-2]:
            raise ValueError("cache position count does not match appended tokens")
        full_keys, full_values = cache_view
        if full_keys.shape != full_values.shape or full_keys.shape[-2] < keys.shape[-2]:
            raise ValueError("native cache returned an invalid logical KV view")
        self._append_calls += 1
        count = int(keys.shape[-2])
        self._appended_tokens += count
        self._tokens_by_layer[layer_idx] = self._tokens_by_layer.get(layer_idx, 0) + count

    def acquire(self, layer_idx, group_idx, keys, values, positions=None):
        if not self._active or self._cache is None:
            raise RuntimeError("KV lease requested without a live model cache")
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[1] != 1:
            raise ValueError("one lease must contain one KV group as [batch, 1, tokens, dim]")
        if positions is not None and positions.numel() != keys.shape[-2]:
            raise ValueError("logical row positions do not match the leased KV rows")
        device = keys.device
        placement = "cuda HBM" if device.type == "cuda" else "ordinary CPU memory"
        logical_bytes = (keys.numel() * keys.element_size()) + (
            values.numel() * values.element_size()
        )
        self._outstanding += 1
        self._leases_issued += 1

        def released(completion):
            if not self._active or self._outstanding < 1:
                raise RuntimeError("KV lease outlived its request")
            self._outstanding -= 1
            self._leases_released += 1

        return _BorrowedKVLease(
            keys=keys,
            values=values,
            positions=positions,
            token_ids=None,
            placement=placement,
            logical_bytes=logical_bytes,
            transfer_bytes=0,
            _owner=self._cache,
            _on_release=released,
        )

    def snapshot(self) -> dict[str, Any]:
        first_layer = None
        if self._cache is not None and self._cache.layers:
            first_layer = self._cache.layers[0].keys
        owner_placement = (
            "cuda HBM" if first_layer is not None and first_layer.is_cuda
            else "ordinary CPU memory" if first_layer is not None
            else "not allocated"
        )
        return {
            "adapter": self.name,
            "request_id": self._request_id,
            "append_calls": self._append_calls,
            "appended_tokens": self._appended_tokens,
            "appended_tokens_by_layer": dict(self._tokens_by_layer),
            "leases_issued": self._leases_issued,
            "leases_released": self._leases_released,
            "outstanding_leases": self._outstanding,
            "owner_placement": owner_placement,
        }

    def end_request(self) -> None:
        if not self._active:
            return
        if self._outstanding:
            raise RuntimeError(f"cannot release cache with {self._outstanding} live KV leases")
        self._cache = None
        self._active = False
