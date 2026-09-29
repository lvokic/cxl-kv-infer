"""Shared greedy decode loop for model adapters implementing ``ModelAdapter``."""

import hashlib
import time

import numpy as np
import torch

from .interfaces import ModelAdapter


class GenerationResult:
    def __init__(self, record, logits=None):
        self.record = record
        self.logits = logits


@torch.inference_mode()
def greedy_generate(adapter: ModelAdapter, prompt, *, request_id="request-0", capture_dir=None, save_logits=False):
    started = time.perf_counter()
    input_ids = adapter.encode(prompt)
    adapter.synchronize()
    tokenize_seconds = time.perf_counter() - started
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    adapter.begin_request(request_id, prompt_hash, capture_dir)
    generated, samples, logits_saved = [], [], []
    cache = None
    overall = time.perf_counter()
    try:
        for index in range(adapter.config.max_new_tokens):
            adapter.synchronize()
            start = time.perf_counter()
            if index == 0:
                step_result = adapter.prefill(input_ids)
            else:
                token = torch.tensor([[generated[-1]]], device=adapter.device)
                step_result = adapter.decode(
                    token, cache, index - 1, input_ids.shape[1] + index - 1
                )
            cache = step_result.cache
            last_logits = step_result.logits[0, -1]
            if not bool(torch.isfinite(last_logits).all()):
                raise RuntimeError("model produced non-finite vocabulary logits")
            next_token = int(last_logits.argmax().item())
            if save_logits:
                logits_saved.append(last_logits.float().cpu().numpy().copy())
            adapter.synchronize()
            elapsed = time.perf_counter() - start
            generated.append(next_token)
            samples.append(
                {
                    "generated_index": index,
                    "phase": "prefill" if index == 0 else "decode",
                    "wall_seconds": elapsed,
                    "token_id": next_token,
                }
            )
            if adapter.config.stop_on_eos and next_token in adapter.eos_token_ids():
                break
        adapter.synchronize()
        generation_seconds = time.perf_counter() - overall
        metadata = adapter.inference_metadata()
        record = {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "input_token_ids": input_ids[0].cpu().tolist(),
            "input_tokens": input_ids.shape[1],
            "generated_token_ids": generated,
            "generated_tokens": len(generated),
            "generated_text": adapter.tokenizer.decode(generated, skip_special_tokens=True),
            "stop_reason": "eos" if adapter.config.stop_on_eos and generated[-1] in adapter.eos_token_ids() else "max_new_tokens",
            "sampling": "greedy",
            "backend": adapter.config.backend,
            "selector": adapter.config.selector,
            "samples": samples,
            "timing": {
                "scope": "diagnostic synchronized single request, not a throughput benchmark",
                "tokenize_and_input_transfer_seconds": tokenize_seconds,
                "prefill_and_first_token_seconds": samples[0]["wall_seconds"],
                "decode_seconds": sum(sample["wall_seconds"] for sample in samples[1:]),
                "generation_wall_seconds": generation_seconds,
                "capture_included": adapter.config.capture,
                "vocabulary_logits_copy_included": save_logits,
                "cold_jit_may_be_included": adapter.config.backend == "flashinfer",
            },
            **metadata,
        }
        return GenerationResult(record, np.stack(logits_saved) if save_logits else None)
    finally:
        adapter.end_request()
