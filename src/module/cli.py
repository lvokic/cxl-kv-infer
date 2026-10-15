"""Load a Llama-family model and run single-request end-to-end decode experiments."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .artifacts import ROOT, record_manifest, write_json
from .config import InferenceConfig, parse_ids


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Hugging Face model ID or local safetensors directory")
    p.add_argument("--revision", help="remote commit/tag (resolved to a commit) or local provenance label")
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts-jsonl", type=Path, help='records with a string "prompt" field')
    p.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths are refused")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--backend", choices=("sdpa", "flashinfer", "reference"), default="sdpa")
    p.add_argument(
        "--selector",
        choices=("full", "exact-topk", "quest", "retr-attn", "retr_attn"),
        default="full",
    )
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--sink-tokens", type=int, default=4)
    p.add_argument("--window-tokens", type=int, default=8)
    p.add_argument("--quest-chunk-size", type=int, default=16)
    p.add_argument("--retrieval-sink-tokens", type=int, default=128)
    p.add_argument("--retrieval-recent-tokens", type=int, default=512)
    p.add_argument("--layers", type=parse_ids, default=(0,), help="comma-separated layer IDs, or all")
    p.add_argument("--groups", type=parse_ids, default=(0,), help="comma-separated KV group IDs, or all")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--capture-step", type=int, default=0, help="0 is the first decode forward after prefill")
    p.add_argument("--cache-dir", default="data/models")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    p.add_argument("--cpu-threads", type=int, help="explicit PyTorch CPU thread limit")
    return p


def main():
    args = parser().parse_args()
    values = vars(args).copy()
    for key in ("prompt", "prompts_jsonl", "output_dir", "raw_prompt", "ignore_eos"):
        values.pop(key)
    config = InferenceConfig(**values, chat=not args.raw_prompt, stop_on_eos=not args.ignore_eos)
    try:
        config.validate()
        if args.prompt is not None:
            prompts = [args.prompt]
        else:
            prompts = [json.loads(line)["prompt"] for line in args.prompts_jsonl.read_text().splitlines() if line.strip()]
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("provide at least one nonempty string prompt")
        run = args.output_dir.resolve()
        run.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(run / "config.json", asdict(config))
    record_manifest(run)
    write_json(run / "prompts.json", prompts)
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/inspect_platform.py"), "--output", str(run / "platform.json")], check=True)
        from .model import TransformersEngine
        engine = TransformersEngine(config)
        write_json(run / "model.json", engine.model_info)
        with (run / "requests.jsonl").open("x", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                result = engine.generate(prompt, request_id=f"request-{i:04d}",
                    capture_dir=run / "captures" / f"request-{i:04d}" if config.capture else None)
                f.write(json.dumps(result.record, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                print(result.record["generated_text"], flush=True)
        write_json(run / "summary.json", {"status": "completed", "requests": len(prompts),
            "model_resolve_and_hash_seconds": engine.resolve_seconds, "model_load_seconds": engine.load_seconds,
            "scope": "single-device end-to-end diagnostic inference", "model_quality_evaluated": False,
            "cxl_allocation": False, "config": asdict(config)})
        print(f"Artifacts: {run}")
        return 0
    except Exception as exc:
        (run / "exception.txt").write_text(traceback.format_exc())
        write_json(run / "summary.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
