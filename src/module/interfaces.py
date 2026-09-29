"""Framework-neutral contracts for model, KV-cache, selection and attention code.

The contracts intentionally use ``Any`` for tensor and owner types. Importing them
must not import PyTorch, Transformers, FlashInfer, QUEST or RetroInfer.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ModelStep:
    """One model forward: next-token logits and the live cache owner."""

    logits: Any
    cache: Any


@dataclass(frozen=True)
class AttentionResult:
    """Executor output and an optional completion token for asynchronous readers."""

    output: Any
    completion: Any = None


class KVCacheLease(Protocol):
    """Borrowed logical KV rows. ``rows`` index this lease, not physical pages."""

    keys: Any
    values: Any
    positions: Any
    token_ids: Any
    placement: str
    logical_bytes: int
    transfer_bytes: int

    def release(self, completion: Any = None) -> None:
        """Return ownership after consumers finish; completion can be a device event."""


class KVCacheLifecycle(Protocol):
    """Per-request cache owner and lease factory used by model adapters."""

    def begin_request(self, request_id: str) -> None: ...

    def new_cache(self, model_config: Any) -> Any:
        """Create the native cache object that the model appends into."""
        ...

    def observe_append(
        self,
        layer_idx: int,
        keys: Any,
        values: Any,
        positions: Any,
        cache_view: Any,
    ) -> None: ...

    def acquire(
        self,
        layer_idx: int,
        group_idx: int,
        keys: Any,
        values: Any,
        positions: Any,
    ) -> KVCacheLease: ...

    def end_request(self) -> None: ...


@dataclass(frozen=True)
class AttentionRequest:
    """One group or whole-layer attention operation."""

    query: Any
    keys: Any
    values: Any
    scale: float
    layer_idx: int
    group_idx: int | None
    rows: Sequence[Any] | None
    lease: KVCacheLease | None = None
    module: Any = None
    attention_mask: Any = None
    dropout: float = 0.0
    backend_kwargs: dict[str, Any] | None = None
    trace: Any = None


class ModelAdapter(Protocol):
    """Model-facing operations consumed by the shared greedy generation loop."""

    config: Any
    device: Any
    tokenizer: Any

    def encode(self, prompt: str) -> Any: ...

    def begin_request(self, request_id: str, prompt_hash: str, capture_dir: Any) -> None: ...

    def prefill(self, input_ids: Any) -> ModelStep: ...

    def decode(self, input_token: Any, cache: Any, step: int, position: int) -> ModelStep: ...

    def eos_token_ids(self) -> set[int]: ...

    def synchronize(self) -> None: ...

    def inference_metadata(self) -> dict[str, Any]: ...

    def end_request(self) -> None: ...


class TokenSelector(Protocol):
    """Produce independent per-query-head row IDs into one KV lease."""

    def select(self, query: Any, lease: KVCacheLease, scale: float) -> Sequence[Any]: ...


class AttentionExecutor(Protocol):
    """Execute the given mask exactly; any async cache use is reflected in completion."""

    def execute(self, request: AttentionRequest) -> AttentionResult: ...
