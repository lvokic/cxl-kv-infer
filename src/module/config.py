"""Configuration for the single-device inference experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    model: str
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    backend: str = "sdpa"
    selector: str = "full"
    top_k: int = 16
    sink_tokens: int = 4
    window_tokens: int = 8
    quest_chunk_size: int = 16
    retrieval_sink_tokens: int = 128
    retrieval_recent_tokens: int = 512
    layers: tuple[int, ...] | None = (0,)
    groups: tuple[int, ...] | None = (0,)
    max_new_tokens: int = 16
    max_input_tokens: int = 2048
    chat: bool = True
    stop_on_eos: bool = True
    capture: bool = False
    capture_step: int = 0
    cache_dir: str = "data/models"
    local_files_only: bool = False
    gpu_memory_fraction: float = 0.5

    def validate(self):
        if not self.model:
            raise ValueError("model ID or local directory is required")
        if self.backend not in ("sdpa", "flashinfer", "reference"):
            raise ValueError("backend must be sdpa, flashinfer, or reference")
        if self.selector not in (
            "full",
            "exact-topk",
            "quest",
            "retr-attn",
            "retr_attn",
        ):
            raise ValueError("selector must be full, exact-topk, quest, or retr-attn/retr_attn")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported model dtype")
        if self.device != "cpu" and not self.device.startswith("cuda:"):
            raise ValueError("use cpu or an explicit cuda:N device")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("the CPU inference baseline requires float32")
        if self.backend == "flashinfer" and (self.device == "cpu" or self.dtype == "float32"):
            raise ValueError("FlashInfer requires CUDA and bfloat16/float16; no CPU fallback")
        if min(
            self.top_k,
            self.sink_tokens,
            self.window_tokens,
            self.retrieval_sink_tokens,
            self.retrieval_recent_tokens,
            self.capture_step,
        ) < 0:
            raise ValueError("selection budgets and capture step must be nonnegative")
        if self.quest_chunk_size < 1:
            raise ValueError("quest_chunk_size must be positive")
        if min(self.max_new_tokens, self.max_input_tokens) < 1:
            raise ValueError("token limits must be positive")
        if self.capture and self.capture_step >= self.max_new_tokens - 1:
            raise ValueError("capture_step must be reached by a decode forward; increase max_new_tokens")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0,1]")
        for indices in (self.layers, self.groups):
            if indices is not None and (not indices or min(indices) < 0 or len(set(indices)) != len(indices)):
                raise ValueError("layers/groups must contain unique nonnegative IDs, or all")


def parse_ids(value):
    return None if value == "all" else tuple(int(x) for x in value.split(","))
