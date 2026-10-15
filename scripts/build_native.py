#!/usr/bin/env python3
"""Build the optional NUMA boundary with bounded CPU parallelism; run no tests."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

# Also works before installing the project entry points.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from module.artifacts import record_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4 or Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        parser.error('use this repository .venv and between 1 and 4 build jobs')
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    record_manifest(run)
    build = ROOT / 'build/native'
    try:
        pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
        commands = [
            ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', f'-DPython3_EXECUTABLE={sys.executable}',
             f'-Dpybind11_DIR={pybind}'],
            ['cmake', '--build', str(build), '--parallel', str(args.jobs)],
        ]
        write_json(run / 'commands.json', commands)
        with (run / 'build.log').open('x') as log:
            for command in commands:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        tools = {}
        for command in (['cmake', '--version'], ['g++', '--version'],
                        ['dpkg-query', '-W', 'libnuma1', 'libnuma-dev']):
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            tools[command[0]] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        from module.artifacts import digest
        binaries = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'src/storage').glob('_numa*.so')}
        if not binaries:
            raise RuntimeError('build did not produce a NUMA extension')
        write_json(run / 'build.json', {'status': 'built', 'tools': tools,
                   'pybind11': importlib.metadata.version('pybind11'), 'binary_sha256': binaries,
                   'runtime_allocation_executed': False, 'tests_executed': False})
        for name in ('CMakeCache.txt', 'compile_commands.json'):
            (run / name).write_bytes((build / name).read_bytes())
        print(f'Built NUMA bridge. Build record: {run}')
        return 0
    except Exception as exc:
        write_json(run / 'build.json', {'status': 'failed', 'error': str(exc), 'tests_executed': False})
        print(f'Build failed; see {run / "build.log"}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
