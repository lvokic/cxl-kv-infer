"""Exclusive run artifacts and input provenance; no model imports."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def source_manifest():
    files = [p for p in (ROOT / "src").rglob("*") if p.is_file() and
             p.suffix in (".py", ".cpp", ".h", ".hpp", ".cu", ".cuh")]
    files += [p for p in (ROOT / "requirements").iterdir() if p.suffix in (".in", ".lock")]
    files += [p for p in (ROOT / "scripts").iterdir() if p.suffix in (".py", ".sh")]
    files += [p for p in (ROOT / "configs").iterdir() if p.suffix in (".json", ".jsonl")]
    files += [ROOT / "pyproject.toml", ROOT / "CMakeLists.txt"]
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
        "native_binary_sha256": {str(p.relative_to(ROOT)): digest(p)
                                  for p in (ROOT / "src").rglob("*.so")},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "launch_environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDACXX", "TORCH_CUDA_ARCH_LIST",
            "FLASHINFER_CUDA_ARCH_LIST", "MAX_JOBS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_DYNAMIC", "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "LIBRARY_PATH", "LD_LIBRARY_PATH")},
    }


def record_manifest(run):
    manifest = source_manifest()
    write_json(Path(run) / "manifest.json", manifest)
    # A dirty tree's commit alone cannot reproduce the code used in an experiment.
    with zipfile.ZipFile(Path(run) / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in manifest["source_sha256"]:
            archive.write(ROOT / relative, relative)
