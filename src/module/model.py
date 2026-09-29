"""Transformers model loading, full prefill and explicit cached greedy decode."""

import hashlib
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .artifacts import digest
from .attention import DecodeController, register_attention
from .config import InferenceConfig
from .generation import GenerationResult, greedy_generate
from .interfaces import ModelStep


def resolve_model(config):
    local = Path(config.model).expanduser()
    if local.is_dir():
        path, revision = local.resolve(), config.revision
    else:
        from huggingface_hub import HfApi, snapshot_download
        revision = config.revision
        if not config.local_files_only and not (revision and re.fullmatch(r"[0-9a-f]{40}", revision)):
            revision = HfApi().model_info(config.model, revision=revision or "main").sha
        path = Path(snapshot_download(config.model, revision=revision or "main", cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "merges.txt", "vocab.txt"],
            max_workers=2)).resolve()
        revision = path.name
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise ValueError("model directory must contain config.json and safetensors weights")
    files = sorted(p for p in path.iterdir() if p.is_file())
    hashes = {p.name: digest(p) for p in files}
    content_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    info = {"model_id": config.model, "revision": revision or "local-sha256:" + content_hash,
            "snapshot_path": str(path), "file_sha256": hashes, "content_sha256": content_hash,
            "config_sha256": hashes["config.json"], "local_directory": local.is_dir()}
    return path, info


class TransformersEngine:
    """One model on one explicit device. All KV stays in its native model cache."""

    def __init__(self, config: InferenceConfig, *, cache_lifecycle=None, executors=None):
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            torch.cuda.set_device(self.device)
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        # Dense top-k must use the recorded FP32 scorer rather than TF32 products.
        torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        path, self.model_info = resolve_model(config)
        self.resolve_seconds = time.perf_counter() - started
        architecture = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if architecture.model_type != "llama":
            raise ValueError("initial inference/capture integration supports Transformers Llama-family models only")
        rope = getattr(architecture, "rope_scaling", None)
        if rope and rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear", "llama3"):
            raise ValueError("this trace contract has not validated dynamic or other RoPE variants")
        if getattr(architecture, "sliding_window", None) is not None:
            raise ValueError("sliding-window score masks are outside this initial integration")
        for selected, size, name in ((config.layers, architecture.num_hidden_layers, "layer"),
                                     (config.groups, architecture.num_key_value_heads, "KV group")):
            if selected is not None and max(selected) >= size:
                raise ValueError(f"selected {name} is outside the model")
        if config.backend == "flashinfer" and (architecture.hidden_size // architecture.num_attention_heads) not in (64, 128):
            raise ValueError("FlashInfer integration supports head_dim 64/128")
        self.model_info["position_transform"] = {"kind": "post_rope", "implementation": "transformers.llama",
            "rope_theta": architecture.rope_theta, "rope_scaling": rope}
        self.model_info["architecture"] = {"model_type": architecture.model_type,
            "layers": architecture.num_hidden_layers, "query_heads": architecture.num_attention_heads,
            "kv_heads": architecture.num_key_value_heads, "hidden_size": architecture.hidden_size,
            "max_position_embeddings": architecture.max_position_embeddings}
        self.model_info["runtime"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled()}
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        register_attention()
        self.controller = DecodeController(
            config, self.model_info, cache_lifecycle=cache_lifecycle, executors=executors
        )
        started = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=getattr(torch, config.dtype), attn_implementation="cxl_kv_decode").to(self.device).eval()
        for layer in self.model.model.layers:
            layer.self_attn._cxl_decode_controller = self.controller
        self.synchronize()
        self.load_seconds = time.perf_counter() - started

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def encode(self, prompt):
        if self.config.chat:
            if not self.tokenizer.chat_template:
                raise ValueError("tokenizer has no chat template; select raw prompt mode")
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] < 1 or ids.shape[1] > self.config.max_input_tokens:
            raise ValueError("prompt token count is empty or exceeds max_input_tokens; no silent truncation")
        if ids.shape[1] + self.config.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus requested generation exceeds the model position limit")
        return ids.to(self.device)

    def prefill(self, input_ids):
        self.controller.phase = "prefill"
        positions = torch.arange(input_ids.shape[1], device=self.device)
        cache = self.controller.cache_lifecycle.new_cache(self.model.config)
        output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            position_ids=positions[None], cache_position=positions, past_key_values=cache,
            use_cache=True, logits_to_keep=1)
        return ModelStep(output.logits, output.past_key_values)

    def decode(self, input_token, cache, step, position):
        self.controller.phase = "decode"
        self.controller.step, self.controller.query_position = step, position
        output = self.model(input_ids=input_token, past_key_values=cache,
            attention_mask=torch.ones((1, position + 1), dtype=torch.long, device=self.device),
            position_ids=torch.tensor([[position]], device=self.device),
            cache_position=torch.tensor([position], device=self.device), use_cache=True, logits_to_keep=1)
        return ModelStep(output.logits, output.past_key_values)

    @torch.inference_mode()
    def generate(self, prompt, *, request_id="request-0", capture_dir=None, save_logits=False):
        return greedy_generate(self, prompt, request_id=request_id,
            capture_dir=capture_dir, save_logits=save_logits)

    def begin_request(self, request_id, prompt_hash, capture_dir):
        self.controller.begin_request(request_id, prompt_hash, capture_dir)

    def end_request(self):
        self.controller.end_request()

    def eos_token_ids(self):
        eos = self.model.generation_config.eos_token_id
        return set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    def inference_metadata(self):
        return {
            "route_counts": dict(self.controller.routes),
            "selection_stats": self.controller.selection_stats,
            "captures": self.controller.captures,
            "cache_lifecycle": self.controller.cache_lifecycle.snapshot(),
            "placement": "model weights and KV on " + str(self.device),
            "cxl_allocation": False,
            "torch_cuda_memory": (
                {
                    "scope": "allocator peaks since model construction; excludes other processes and non-PyTorch allocations",
                    "allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                }
                if self.device.type == "cuda"
                else None
            ),
        }
