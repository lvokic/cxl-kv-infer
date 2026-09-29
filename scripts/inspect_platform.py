#!/usr/bin/env python3
"""Read-only platform snapshot; standard library only, no benchmarks or migration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess


def command(argv: list[str]) -> dict:
    if shutil.which(argv[0]) is None:
        return {"argv": argv, "available": False}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "error": "timeout"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-bdf", default="0000:8b:00.0")
    args = parser.parse_args()
    pci = Path("/sys/bus/pci/devices") / args.gpu_bdf.lower()
    snapshot = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": platform.release(), "python": platform.python_version(),
        "gpu_bdf": args.gpu_bdf,
        "pci_sysfs": {
            name: (pci / name).read_text().strip() if (pci / name).exists() else None
            for name in ("numa_node", "current_link_speed", "current_link_width")
        },
        "commands": [command(argv) for argv in (
            ["lscpu"], ["numactl", "--hardware"],
            ["nvidia-smi", "--query-gpu=name,pci.bus_id,memory.total,memory.used,driver_version", "--format=csv"],
            ["nvidia-smi", "topo", "-m"], ["nvcc", "--version"],
        )],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
