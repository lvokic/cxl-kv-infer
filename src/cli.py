"""Offline trace fixture, inspection, and one reference execution.

This CLI makes no CXL placement or performance claims and runs no GPU by default.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from reference import selected_attention
from kv_trace import GroupTrace, synthetic_trace


def describe(trace: GroupTrace) -> dict:
    unique = np.unique(trace.indices).size
    return {
        "metadata": trace.metadata,
        "shape": {"heads": trace.num_heads, "tokens": len(trace.k), "dim": trace.head_dim},
        "selected_per_head": np.diff(trace.indptr).tolist(),
        "selected_entries": int(trace.indices.size),
        "unique_selected_tokens": int(unique),
        "sharing_ratio": float(trace.indices.size / unique) if unique else None,
        "selected_kv_payload_bytes_at_bf16": int(unique * trace.head_dim * 4),
        "payload_note": "analytical BF16 payload; not measured CXL or PCIe traffic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("make-synthetic", help="write a small synthetic NPZ fixture")
    synthetic.add_argument("--output", type=Path, required=True)
    for name, default in (("heads", 4), ("tokens", 128), ("dim", 128), ("selected", 16), ("seed", 0)):
        synthetic.add_argument(f"--{name}", type=int, default=default)
    inspect = sub.add_parser("inspect", help="describe a trace; no model or GPU execution")
    inspect.add_argument("trace", type=Path)
    replay = sub.add_parser("replay", help="one reference or GPU-adapter execution; no benchmark")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--backend", choices=("reference", "flashinfer"), default="reference")
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-synthetic":
            trace = synthetic_trace(heads=args.heads, tokens=args.tokens, dim=args.dim,
                                    selected=args.selected, seed=args.seed)
            trace.save(args.output)
            print(json.dumps(describe(trace), indent=2, ensure_ascii=False, allow_nan=False))
            return
        trace = GroupTrace.load(args.trace)
        if args.command == "inspect":
            print(json.dumps(describe(trace), indent=2, ensure_ascii=False, allow_nan=False))
            return
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        if args.output.suffix != ".npz":
            raise ValueError("output path must end in .npz")
        if args.backend == "reference":
            state = selected_attention(trace)
        else:
            from backends.flashinfer import selected_attention_flashinfer
            state = selected_attention_flashinfer(trace, device=args.device, dtype=args.dtype)
        digest = hashlib.sha256()
        with args.trace.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        summary = {
            **describe(trace), "backend": args.backend, "trace_sha256": digest.hexdigest(),
            "lse_base": "e", "timing_reported": False, "cxl_placement": "not implemented",
            "backend_dtype": "float64" if args.backend == "reference" else args.dtype,
            "device": "cpu" if args.backend == "reference" else args.device,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            np.savez(handle, output=state.output, lse=state.lse,
                     summary_json=np.asarray(json.dumps(summary, allow_nan=False)))
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
