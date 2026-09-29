#!/usr/bin/env python3
"""End-to-end model integration checks; not a task-quality or speed benchmark."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

from module.artifacts import ROOT, digest, record_manifest, write_json
from module.config import InferenceConfig
from module.model import TransformersEngine
from reference import selected_attention
from kv_trace import GroupTrace

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PROMPT = "A library has books on science, history, and art. Explain in one short sentence why people read books."
# Fixed before running. Deep-model BF16 logits have a different budget from a single attention output.
LOGIT_ATOL, LOGIT_RTOL, LOGIT_REL_L2 = 0.1, 0.01, 0.005


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float16")
    args = p.parse_args()
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    base = InferenceConfig(model=args.model, revision=args.revision, dtype=args.dtype, local_files_only=True,
        max_new_tokens=5, capture=True, stop_on_eos=False, gpu_memory_fraction=0.1)
    write_json(run / "config.json", {"inference": asdict(base), "prompt": PROMPT,
        "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL, "logit_relative_l2": LOGIT_REL_L2,
        "script_sha256": digest(Path(__file__))})
    record_manifest(run)
    rows = []

    def check(name, passed, **details):
        row = {"check": name, "passed": bool(passed), **details}
        rows.append(row)
        with (run / "checks.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print("PASS" if passed else "FAIL", name, flush=True)

    def compare(name, actual, expected, strict=False):
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        atol, rtol = (1e-6, 1e-6) if strict else (LOGIT_ATOL, LOGIT_RTOL)
        ratio = float(np.max(delta / (atol + rtol * np.abs(expected))))
        rel_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))
        check(name, np.isfinite(actual).all() and ratio <= 1 and (strict or rel_l2 <= LOGIT_REL_L2),
              max_abs=float(delta.max()), max_tolerance_ratio=ratio, relative_l2=rel_l2)

    def configure(engine, config):
        config.validate()
        engine.config = config
        engine.controller.configure(config)
        engine.model.set_attn_implementation("cxl_kv_decode")

    @torch.inference_mode()
    def forced(engine, config, continuation, name):
        configure(engine, config)
        ids = engine.encode(PROMPT)
        capture = run / name / "captures"
        engine.controller.begin_request(name, hashlib.sha256(PROMPT.encode()).hexdigest(), capture, "teacher_forced_diagnostic")
        try:
            outputs, saved = engine.prefill(ids), []
            saved.append(outputs.logits[0, -1].float().cpu().numpy().copy())
            cache = outputs.cache
            for step, token in enumerate(continuation[:-1]):
                outputs = engine.decode(torch.tensor([[token]], device=engine.device), cache, step, ids.shape[1] + step)
                cache = outputs.cache
                saved.append(outputs.logits[0, -1].float().cpu().numpy().copy())
            actual = np.stack(saved)
            write_json(run / name / "routing.json", {"routes": dict(engine.controller.routes),
                       "selection": engine.controller.selection_stats, "cache_lifecycle": engine.controller.cache_lifecycle.snapshot(),
                       "config": asdict(config)})
            with (run / name / "logits.npy").open("xb") as f:
                np.save(f, actual, allow_pickle=False)
            check(name + "/backend_executed", engine.controller.routes["decode_target_" + config.backend] > 0)
            return actual, capture
        finally:
            engine.end_request()

    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/inspect_platform.py"), "--output", str(run / "platform.json")], check=True)
        engine = TransformersEngine(base)
        write_json(run / "model.json", engine.model_info)
        result = engine.generate(PROMPT, capture_dir=run / "free-full" / "captures", save_logits=True)
        write_json(run / "free-full" / "generation.json", result.record)
        tokens = result.record["generated_token_ids"]
        engine.model.set_attn_implementation("sdpa")
        generation = engine.model.generation_config.from_dict(engine.model.generation_config.to_dict())
        generation.do_sample, generation.eos_token_id, generation.max_new_tokens = False, None, base.max_new_tokens
        ids = engine.encode(PROMPT)
        with torch.inference_mode():
            native = engine.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation,
                return_dict_in_generate=True, output_logits=True)
        native_ids = native.sequences[0, ids.shape[1]:].cpu().tolist()
        native_logits = torch.stack([x[0] for x in native.logits]).float().cpu().numpy()
        check("greedy/native_generate_token_ids", tokens == native_ids, actual=tokens, expected=native_ids)
        compare("greedy/native_generate_logits", result.logits, native_logits, strict=True)
        with (run / "native-generate.npz").open("xb") as f:
            np.savez(f, logits=native_logits, generated_ids=native_ids)
        # Full-prefix, no-cache forward anchors cache_position / append behavior independently.
        uncached = []
        with torch.inference_mode():
            for step in range(len(tokens)):
                prefix = torch.cat((ids, torch.tensor([tokens[:step]], dtype=torch.long, device=engine.device)), dim=1)
                out = engine.model(input_ids=prefix, use_cache=False, logits_to_keep=1)
                uncached.append(out.logits[0, -1].float().cpu().numpy())
        compare("cache/cached_vs_full_prefix_logits", result.logits, np.stack(uncached))
        alternative = "reference" if base.dtype == "float32" else "flashinfer"
        alt, alt_capture = forced(engine, replace(base, backend=alternative), tokens, "forced-full-" + alternative)
        compare("full/" + alternative + "_vs_sdpa_logits", alt, native_logits)
        check("full/" + alternative + "_top1", np.array_equal(alt.argmax(-1), native_logits.argmax(-1)))
        if alternative != "reference":
            ref, _ = forced(engine, replace(base, backend="reference"), tokens, "forced-full-reference")
            compare("full/reference_vs_sdpa_logits", ref, native_logits)
        sparse = replace(base, selector="exact-topk", top_k=2, sink_tokens=1, window_tokens=1)
        a, a_capture = forced(engine, sparse, tokens, "forced-topk-sdpa")
        b, b_capture = forced(engine, replace(sparse, backend=alternative), tokens, "forced-topk-" + alternative)
        compare("same_selection/topk_" + alternative + "_vs_sdpa_logits", b, a)
        left = GroupTrace.load(next(a_capture.glob("*.npz")))
        right = GroupTrace.load(next(b_capture.glob("*.npz")))
        check("same_selection/csr_and_qkv", all(np.array_equal(getattr(left, f), getattr(right, f))
              for f in ("q", "k", "v", "indices", "indptr", "positions")))
        check("selection/sparse_and_mandatory", all(len(left.selected(h)) == 4 and 0 in left.selected(h)
              and len(left.k) - 1 in left.selected(h) for h in range(left.num_heads)) and len(left.k) > 4)
        # Existing offline numerical reference consumes a real, post-RoPE GroupTrace.
        for label, directory in (("full", alt_capture), ("topk", b_capture)):
            path = next(directory.glob("*.npz"))
            trace = GroupTrace.load(path)
            actual = np.load(path.with_suffix(".attention.npy"), allow_pickle=False)
            expected = selected_attention(trace).output
            err = np.abs(actual - expected)
            atol, rtol = {"bfloat16": (2e-3, 8e-3), "float16": (2e-4, 1e-3), "float32": (1e-6, 1e-5)}[base.dtype]
            check("capture/" + label + "/attention_vs_fp64", bool(np.all(err <= atol + rtol * np.abs(expected))),
                  max_abs=float(err.max()))
            check("capture/" + label + "/provenance", trace.metadata["origin"] == "model"
                  and trace.metadata["model_revision"] == engine.model_info["revision"]
                  and trace.metadata["query_position"] == len(trace.k) - 1
                  and trace.metadata["q_head_ids"] == list(range(trace.num_heads))
                  and trace.metadata["capture_mode"] == "teacher_forced_diagnostic")
        # Replay the same real trace through the existing public CLI.
        trace_path = next(alt_capture.glob("*.npz"))
        replay_path = run / "real-trace-reference.npz"
        replay = subprocess.run([sys.executable, str(ROOT / "src/cli.py"), "replay", str(trace_path),
                                 "--output", str(replay_path)], capture_output=True, text=True)
        (run / "real-trace-replay.log").write_text(replay.stdout + replay.stderr)
        check("capture/existing_replay_cli", replay.returncode == 0)
        # Free generation also runs through the selected path and starts a fresh cache.
        configure(engine, replace(sparse, backend=alternative, capture=False))
        original_forward = engine.controller.forward
        unchanged_heads = []

        def inspect_forward(module, q, k, v, mask, **kwargs):
            output = original_forward(module, q, k, v, mask, **kwargs)
            if engine.controller.phase == "decode" and module.layer_idx == 0:
                from transformers.integrations.sdpa_attention import sdpa_attention_forward
                baseline, _ = sdpa_attention_forward(module, q, k, v, mask, **kwargs)
                heads_per_group = q.shape[1] // k.shape[1]
                unchanged_heads.append(torch.equal(output[0][:, :, heads_per_group:], baseline[:, :, heads_per_group:]))
            return output

        engine.controller.forward = inspect_forward
        try:
            sparse_result = engine.generate(PROMPT)
        finally:
            engine.controller.forward = original_forward
        write_json(run / ("free-topk-" + alternative + ".json"), sparse_result.record)
        check("generation/sparse_free_decode", sparse_result.record["generated_tokens"] == base.max_new_tokens
              and sparse_result.record["route_counts"].get("decode_target_" + alternative, 0) == base.max_new_tokens - 1)
        check("dispatch/untargeted_heads_identical", len(unchanged_heads) == base.max_new_tokens - 1 and all(unchanged_heads))
    except Exception as exc:
        check("execution_exception", False, exception=type(exc).__name__, message=str(exc))
        (run / "exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    summary = {"passed": all(x["passed"] for x in rows), "checks": len(rows),
               "failures": [x for x in rows if not x["passed"]],
               "scope": "real pretrained model integration; no task-quality claim or CXL benchmark"}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
