#!/usr/bin/env python3
"""Reproducible numerical validation on synthetic traces, never a benchmark."""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from reference import AttentionState, empty_state, merge_states, selected_attention
from kv_trace import GroupTrace, synthetic_trace

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {
    "float64": {"output": [1e-12, 1e-12], "lse": [1e-12, 1e-12]},
    "bfloat16": {"output": [2e-3, 8e-3], "lse": [2e-4, 2e-6]},
    "float16": {"output": [2e-4, 1e-3], "lse": [2e-4, 2e-6]},
}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_arrays(path, **arrays):
    with path.open("xb") as f:
        np.savez(f, **arrays)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(trace, rows):
    return replace(trace, indptr=np.r_[0, np.cumsum([len(r) for r in rows])].astype(np.int64),
                   indices=np.concatenate([np.asarray(r, dtype=np.int64) for r in rows]))


def subset(trace, membership):
    return selection(trace, [trace.indices[lo:hi][membership[lo:hi]]
                            for lo, hi in zip(trace.indptr[:-1], trace.indptr[1:])])


def scalar_oracle(trace):
    """Independent scalar reductions: no NumPy matmul or production merge code."""
    output = np.zeros(trace.q.shape, dtype=np.float64)
    lse = np.full(trace.num_heads, -np.inf)
    for h in range(trace.num_heads):
        ids = trace.selected(h).tolist()
        if not ids:
            continue
        scores = [math.fsum(float(a) * float(b) for a, b in zip(trace.q[h], trace.k[i]))
                  * float(trace.metadata["sm_scale"]) for i in ids]
        maximum = max(scores)
        weights = [math.exp(s - maximum) for s in scores]
        z = math.fsum(weights)
        lse[h] = maximum + math.log(z)
        for d in range(trace.head_dim):
            output[h, d] = math.fsum(w * float(trace.v[i, d]) for w, i in zip(weights, ids)) / z
    return AttentionState(output, lse)


def fixtures():
    """GPU fixtures use binary fractions exactly representable in both target dtypes."""
    cases = []
    base = synthetic_trace(heads=4, tokens=17, dim=128, selected=5, seed=100)
    guard = selection(replace(base, q=np.zeros_like(base.q)), [[0], [1], [], [0, 1]])
    guard.v[0] = 1
    guard.v[1] = -1
    cases.append(("head_mask_guard", guard, True))
    uniform = selection(replace(base, q=np.zeros_like(base.q)), [[0, 2, 4], [2, 4], [], [16]])
    cases.append(("uniform_logits", uniform, True))
    one = selection(base, [[0], [16], [], [7]])
    one = replace(one, metadata={**one.metadata, "sm_scale": 0.75})
    cases.append(("single_token_scale", one, True))
    overlap = selection(base, [[0, 2, 4, 8, 16], [16, 8, 4], [], [1, 7]])
    positions = np.arange(len(base.k), dtype=np.int64)[::-1] * 3
    overlap = replace(overlap, positions=positions,
                      metadata={**overlap.metadata, "query_position": int(positions.max())})
    cases.append(("overlap_unsorted_positions", overlap, True))
    empty = selection(base, [[], [], [], []])
    cases.append(("all_empty", empty, True))
    cases.append(("no_kv_rows", replace(empty, k=base.k[:0], v=base.v[:0],
                                       positions=base.positions[:0]), True))
    extreme = synthetic_trace(heads=4, tokens=9, dim=128, selected=0, seed=101)
    extreme.q[:] = 0
    extreme.q[:, 0] = [1, -1, 1, -1]
    extreme.k[:] = 0
    extreme.k[:, 0] = [1000, 1000, -1000, -1000, 0, 2, -2, 8, -8]
    extreme = selection(replace(extreme, metadata={**extreme.metadata, "sm_scale": 1.0}),
                        [[0, 1, 2], [0, 1], [2, 3], [2, 3, 4]])
    cases.append(("extreme_logits", extreme, True))
    shift = synthetic_trace(heads=4, tokens=9, dim=128, selected=7, seed=102)
    shift.q[:] = 0
    shift.q[:, 0] = 1
    shift.k[:] = 0
    shift.k[:, 0] = np.arange(-8, 10, 2)
    shift = replace(shift, metadata={**shift.metadata, "sm_scale": 1.0})
    cases.append(("shift_base", shift, True))
    shifted = replace(shift, k=shift.k.copy())
    shifted.k[:, 0] += 256
    cases.append(("shift_plus_256", shifted, True))
    for seed in range(24):
        dim = (1, 7, 64, 128)[seed % 4]
        heads = (1, 4, 8)[seed % 3]
        tokens = (1, 17, 129, 257)[(seed // 4) % 4]
        trace = synthetic_trace(heads=heads, tokens=tokens, dim=dim, selected=0, seed=seed)
        rng = np.random.default_rng(9000 + seed)
        rows = [rng.choice(tokens, size=int(rng.integers(0, tokens + 1)), replace=False)
                for _ in range(heads)]
        trace = selection(trace, rows)
        cases.append((f"random_{seed:02d}", trace, dim in (64, 128) and seed < 16))
    for name, trace, gpu in cases:
        trace = replace(trace, metadata={**trace.metadata, "validation_case": name,
                                        "source_dtype": "float32"})
        trace.validate()
        yield name, trace, gpu


class Checks:
    def __init__(self, run):
        self.rows = []
        self.log = (run / "checks.jsonl").open("x", encoding="utf-8")

    def record(self, name, passed, **details):
        row = {"check": name, "passed": bool(passed), **details}
        self.rows.append(row)
        self.log.write(json.dumps(row, allow_nan=False) + "\n")
        self.log.flush()
        if not passed:
            print("FAIL", name, flush=True)

    def state(self, name, actual, expected, precision="float64"):
        for field in ("output", "lse"):
            a, e = getattr(actual, field), getattr(expected, field)
            finite = np.isfinite(e)
            structure = (a.shape == e.shape and np.array_equal(np.isneginf(a), np.isneginf(e))
                         and not np.isnan(a).any() and not np.isposinf(a).any())
            atol, rtol = TOLERANCES[precision][field]
            error = np.abs(a[finite] - e[finite]) if structure else np.array([np.inf])
            ratio = error / (atol + rtol * np.abs(e[finite])) if structure else error
            max_error, max_ratio = float(error.max(initial=0)), float(ratio.max(initial=0))
            self.record(f"{name}/{field}", structure and max_ratio <= 1,
                        precision=precision, max_abs_error=max_error if math.isfinite(max_error) else None,
                        max_tolerance_ratio=max_ratio if math.isfinite(max_ratio) else None)
        empty = np.isneginf(expected.lse)
        self.record(f"{name}/empty_heads", np.array_equal(actual.output[empty], np.zeros_like(actual.output[empty])))

    def rejects(self, name, action, error_type=ValueError, message=None):
        try:
            action()
        except Exception as exc:
            self.record(name, isinstance(exc, error_type) and (message is None or message in str(exc)),
                        exception=type(exc).__name__, message=str(exc))
        else:
            self.record(name, False, message="expected rejection did not occur")


def cpu_checks(run, checks, cases):
    references = {}
    for number, (name, trace, _) in enumerate(cases):
        trace.save(run / "traces" / f"{name}.npz")
        loaded = GroupTrace.load(run / "traces" / f"{name}.npz")
        checks.record(f"{name}/npz_roundtrip", trace.metadata == loaded.metadata and all(
            np.array_equal(getattr(trace, f), getattr(loaded, f))
            for f in ("q", "k", "v", "positions", "indptr", "indices")))
        ref, oracle = selected_attention(loaded), scalar_oracle(loaded)
        references[name] = ref
        checks.state(f"{name}/reference_oracle", ref, oracle)
        labels = np.random.default_rng(4000 + number).integers(0, 3, size=trace.indices.size)
        masks = [labels == i for i in range(4)]  # fourth partition always empty
        checks.record(f"{name}/partition_cover", bool(np.all(np.sum(masks, axis=0) == 1)))
        parts = [selected_attention(trace, mask) for mask in masks]
        for order in (list(range(4)), list(reversed(range(4)))):
            state = empty_state(trace.num_heads, trace.head_dim)
            for i in order:
                state = merge_states(state, parts[i])
            checks.state(f"{name}/merge_{order}", state, ref)
        balanced = merge_states(merge_states(parts[0], parts[1]), merge_states(parts[2], parts[3]))
        checks.state(f"{name}/merge_balanced", balanced, ref)
        save_arrays(run / "outputs" / f"{name}-reference.npz", output=ref.output, lse=ref.lse,
                    oracle_output=oracle.output, oracle_lse=oracle.lse, partition_labels=labels,
                    partial_output=np.stack([p.output for p in parts]), partial_lse=np.stack([p.lse for p in parts]))
    guard = references["head_mask_guard"]
    checks.state("analytic/head_mask", guard, AttentionState(
        np.stack([np.ones(128), -np.ones(128), np.zeros(128), np.zeros(128)]),
        np.array([0, 0, -np.inf, math.log(2)])))
    uniform_trace = next(t for n, t, _ in cases if n == "uniform_logits")
    uniform_o = np.zeros_like(uniform_trace.q, dtype=np.float64)
    for h in (0, 1, 3):
        uniform_o[h] = uniform_trace.v[uniform_trace.selected(h)].astype(np.float64).mean(axis=0)
    checks.state("analytic/uniform", references["uniform_logits"], AttentionState(
        uniform_o, np.array([math.log(3), math.log(2), -np.inf, 0])))
    shifted = references["shift_plus_256"]
    checks.state("analytic/constant_shift", AttentionState(shifted.output, shifted.lse - 256),
                 references["shift_base"])
    base = cases[0][1]
    bad = replace(base, indices=base.indices.copy())
    bad.indices[-1] = bad.indices[-2]
    checks.rejects("contract/duplicate_token", bad.validate, message="duplicate")
    checks.rejects("contract/future_token", replace(base, metadata={**base.metadata, "query_position": -1}).validate,
                   message="future token")
    bad_q = base.q.copy()
    bad_q[0, 0] = np.nan
    checks.rejects("contract/nonfinite_q", replace(base, q=bad_q).validate, message="non-finite")
    checks.rejects("contract/missing_metadata", replace(base, metadata={}).validate, message="missing metadata")
    return references


def gpu_checks(run, checks, cases, references, device):
    import torch
    from backends.flashinfer import selected_attention_flashinfer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requested but CUDA is unavailable; no CPU fallback")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < 2 * 1024 ** 3:
        raise RuntimeError("GPU validation requires at least 2 GiB free memory")
    torch.cuda.set_per_process_memory_fraction(0.05, device)
    torch.cuda.reset_peak_memory_stats(device)
    result = {}
    for dtype in ("bfloat16", "float16"):
        for name, trace, enabled in cases:
            if not enabled:
                continue
            # Inspect exact representability on CPU before asking the adapter to upload.
            checks.record(f"{name}/{dtype}/identical_inputs", all(np.array_equal(
                torch.from_numpy(a).to(getattr(torch, dtype)).float().numpy(), a)
                for a in (trace.q, trace.k, trace.v)))
            actual = selected_attention_flashinfer(trace, device=device, dtype=dtype)
            checks.state(f"{name}/{dtype}/gpu", actual, references[name], dtype)
            result[(dtype, name)] = actual
            save_arrays(run / "outputs" / f"{name}-{dtype}.npz", output=actual.output, lse=actual.lse)
            if name in ("overlap_unsorted_positions", "extreme_logits", "random_15"):
                mask = np.arange(trace.indices.size) % 2 == 0
                parts = [selected_attention_flashinfer(subset(trace, m), device=device, dtype=dtype)
                         for m in (mask, ~mask)]
                merged = merge_states(*parts)
                checks.state(f"{name}/{dtype}/gpu_partial_merge", merged, references[name], dtype)
                save_arrays(run / "outputs" / f"{name}-{dtype}-parts.npz", membership=mask,
                            partial_output=np.stack([p.output for p in parts]),
                            partial_lse=np.stack([p.lse for p in parts]), output=merged.output, lse=merged.lse)
            print("GPU", dtype, name, flush=True)
        a, b = result[(dtype, "shift_base")], result[(dtype, "shift_plus_256")]
        checks.state(f"analytic/{dtype}/constant_shift", AttentionState(b.output, b.lse - 256), a, dtype)
        base = cases[0][1]
        q = base.q.copy()
        q[0, 0] = np.float32(0.1)
        checks.rejects(f"contract/{dtype}/reject_lossy_input", lambda: selected_attention_flashinfer(
            replace(base, q=q), device=device, dtype=dtype), message="would change trace values")
    write_json(run / "gpu_memory.json", {"device": device, "name": torch.cuda.get_device_name(device),
               "total_bytes": total, "free_bytes_before": free, "allocator_fraction_limit": 0.05,
               "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "note": "PyTorch allocator only; excludes some driver/JIT memory; not a performance measurement"})


def cli_checks(run, checks, reference):
    trace = run / "traces" / "head_mask_guard.npz"
    output = run / "outputs" / "cli-reference.npz"
    for command, extra in (("inspect", []), ("replay", ["--output", str(output)])):
        p = subprocess.run([sys.executable, str(ROOT / "src/cli.py"), command, str(trace), *extra],
                           capture_output=True, text=True, check=False)
        (run / f"cli-{command}.txt").write_text(p.stdout + p.stderr)
        checks.record(f"cli/{command}", p.returncode == 0)
    with np.load(output, allow_pickle=False) as arrays:
        checks.state("cli/reference", AttentionState(arrays["output"], arrays["lse"]), reference)
        summary = json.loads(str(arrays["summary_json"].item()))
        checks.record("cli/provenance", summary["trace_sha256"] == sha256(trace) and summary["lse_base"] == "e"
                      and summary["timing_reported"] is False)
    digest = sha256(output)
    p = subprocess.run([sys.executable, str(ROOT / "src/cli.py"), "replay", str(trace), "--output", str(output)],
                       capture_output=True, text=True, check=False)
    checks.record("cli/refuse_overwrite", p.returncode == 2 and sha256(output) == digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "flashinfer"), default="reference")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "traces").mkdir()
    (run / "outputs").mkdir()
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/with-env.sh",
            ROOT / "requirements/core.lock", ROOT / "requirements/gpu-cu130.lock", ROOT / "docs/NUMERICAL_BASELINE.md"]
    write_json(run / "config.json", {"backend": args.backend, "device": args.device,
               "tolerances": TOLERANCES, "input_origin": "synthetic", "model_revision": None,
               "timing_reported": False, "cxl_placement": "not implemented", "gpu_allocator_fraction": 0.05})
    write_json(run / "manifest.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "executable": sys.executable,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
               "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
               "environment": {k: os.environ.get(k) for k in ("CUDA_HOME", "CUDA_PATH", "CUDACXX", "LIBRARY_PATH", "LD_LIBRARY_PATH",
                  "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST", "MAX_JOBS", "OPENBLAS_NUM_THREADS",
                  "OMP_NUM_THREADS", "FLASHINFER_WORKSPACE_BASE", "TORCH_EXTENSIONS_DIR")}})
    checks = Checks(run)
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/inspect_platform.py"), "--output", str(run / "platform.json")], check=True)
        cases = list(fixtures())
        refs = cpu_checks(run, checks, cases)
        cli_checks(run, checks, refs["head_mask_guard"])
        if args.backend == "flashinfer":
            gpu_checks(run, checks, cases, refs, args.device)
    except Exception as exc:
        checks.record("execution_exception", False, exception=type(exc).__name__, message=str(exc))
        (run / "exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        counts = Counter("passed" if r["passed"] else "failed" for r in checks.rows)
        maxima = {}
        for precision in TOLERANCES:
            for field in ("output", "lse"):
                vals = [r["max_abs_error"] for r in checks.rows if r.get("precision") == precision
                        and r["check"].endswith("/" + field) and r.get("max_abs_error") is not None]
                if vals:
                    maxima[f"{precision}/{field}"] = max(vals)
        summary = {"backend": args.backend, "counts": dict(counts), "max_abs_error": maxima,
                   "passed": counts["failed"] == 0, "scope": "synthetic numerical validation only",
                   "torch_imported": "torch" in sys.modules}
        write_json(run / "summary.json", summary)
        write_json(run / "artifact_sha256.json", {str(p.relative_to(run)): sha256(p)
                    for p in sorted(run.rglob("*.npz"))})
        checks.log.close()
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
