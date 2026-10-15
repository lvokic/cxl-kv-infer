"""Prepare recorded experiments; launch local inference only on explicit request."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from index.backends import backend_inventory
from module.artifacts import ROOT, digest, record_manifest, write_json
from .config import ExperimentConfig
from .topology import Topology, worker_affinity


def gpu_status():
    command = ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc), 'devices': []}
    if result.returncode:
        return {'available': False, 'error': result.stderr, 'devices': []}
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        ordinal, name, total, used, utilization = [x.strip() for x in row]
        devices.append({'ordinal': int(ordinal), 'name': name, 'total_bytes': int(total) * 1024**2,
                        'used_bytes': int(used) * 1024**2, 'utilization_percent': int(utilization)})
    return {'available': True, 'devices': devices}


def local_model_info(config):
    path = Path(config.values['model']['path']).expanduser().resolve()
    architecture = json.loads((path / 'config.json').read_text())
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    required = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'model.safetensors.index.json', 'LICENSE', 'USE_POLICY.md'}
    required.update(index['weight_map'].values())
    for name in required:
        if Path(name).name != name or not (path / name).is_file():
            raise ValueError(f'missing or unsafe checkpoint path: {name}')
    if (architecture['model_type'] != 'llama' or architecture['num_hidden_layers'] != 32 or
            architecture['num_attention_heads'] != 32 or architecture['num_key_value_heads'] != 8 or
            architecture['hidden_size'] != 4096):
        raise ValueError('this pilot configuration expects Llama-3.1-8B architecture')
    model = config.values['model']
    if model['max_input_tokens'] + model['max_new_tokens'] > architecture['max_position_embeddings']:
        raise ValueError('requested prompt/generation exceeds the model position limit')
    manifests = [(p, json.loads(p.read_text())) for p in sorted(path.glob('download-manifest-*.json'))]
    manifests = [(p, j) for p, j in manifests if j.get('status') == 'complete' and
                 j.get('canonical_revision') == model['revision']]
    if not manifests:
        raise ValueError('checkpoint needs a completed download manifest matching its fixed canonical revision')
    provenance_path, provenance = manifests[-1]
    for name in required:
        recorded = provenance['files'].get(name)
        if not recorded or not recorded['canonical_match'] or (path / name).stat().st_size != recorded['size_bytes']:
            raise ValueError(f'checkpoint metadata/size does not match download provenance: {name}')
        if not name.endswith('.safetensors') and digest(path / name) != recorded['sha256']:
            raise ValueError(f'checkpoint metadata hash mismatch: {name}')
    kv = 2 * 32 * 8 * 128 * (4 if model['dtype'] == 'float32' else 2) * model['max_input_tokens']
    return {'path': str(path), 'canonical_model_id': provenance['canonical_model_id'],
            'canonical_revision': provenance['canonical_revision'],
            'download_model_id': provenance['download_model_id'], 'download_revision': provenance['download_revision'],
            'download_manifest_sha256': digest(provenance_path), 'architecture': architecture,
            'files': {n: {'bytes': (path / n).stat().st_size, 'recorded_sha256': provenance['files'][n]['sha256']}
                      for n in sorted(required)},
            'preflight_scope': 'current file sizes and metadata hashes; weight SHA-256 from completed download, not rehashed',
            'kv_payload_estimate_bytes': kv, 'estimate_scope': 'full GPU KV at max_input_tokens; excludes workspace and generated tokens'}


def baseline_info():
    result = {}
    for name in ('RetrievalAttention', 'MInference'):
        path = Path.home() / name
        if not (path / '.git').exists():
            result[name] = {'available': False}
            continue
        result[name] = {'path': str(path), 'commit': subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', '-C', str(path), 'status', '--short'], text=True),
            'scope': 'source checkout only; not installed or executed'}
    return result


def prepare(config, run):
    write_json(run / 'config.json', {'profile': config.profile, 'values': config.values})
    subprocess.run([sys.executable, str(ROOT / 'scripts/inspect_platform.py'),
                    '--output', str(run / 'platform.json')], check=True)
    model = local_model_info(config)
    write_json(run / 'model.json', model)
    write_json(run / 'baselines.json', baseline_info())
    dependencies = {}
    for name in ('numpy', 'torch', 'transformers', 'flashinfer-python', 'pybind11'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f'missing dependency {name}; follow docs/DEVELOPMENT.md')
    native = importlib.util.find_spec('storage._numa')
    if native is None:
        raise RuntimeError('NUMA bridge not built; run scripts/build_native.py')
    index_backends = {'selected': backend_inventory(config.values['index']['backend']),
                      'control': backend_inventory(config.values['index']['control']['backend'])}
    write_json(run / 'index-backends.json', index_backends)
    ruler = Path(config.values['benchmarks']['ruler']).expanduser()
    benchmark = {'ruler_path': str(ruler), 'exists': ruler.is_dir(),
                 'generated_data_verified': False,
                 'scope': 'directory/source inventory; dataset generation and quality evaluation not performed'}
    if (ruler / 'README.md').is_file():
        readme = (ruler / 'README.md').read_text()
        benchmark['readme_sha256'] = digest(ruler / 'README.md')
        benchmark['upstream_workflow'] = 'NeMo-Skills' if 'NeMo-Skills' in readme else 'inspect pinned source'
    if (ruler / '.git').is_dir():
        benchmark['commit'] = subprocess.check_output(['git', '-C', str(ruler), 'rev-parse', 'HEAD'], text=True).strip()
        benchmark['git_status'] = subprocess.check_output(['git', '-C', str(ruler), 'status', '--short'], text=True)
    write_json(run / 'benchmarks.json', benchmark)
    status = {'status': 'prepared', 'scope': 'read-only asset/dependency/topology readiness',
              'profile': config.profile, 'placement': config.placement,
              'configured_tiers': config.values['tiers'], 'topology': Topology().snapshot(),
              'dependencies': dependencies, 'native_binary': native.origin,
              'index_backends': index_backends,
              'native_scope': 'binary present; allocation and residency syscalls not executed by prepare',
              'ledger': config.make_ledger().snapshot(), 'gpu': gpu_status(),
              'child_environment': config.child_environment(),
              'model_inference_executed': False, 'index_operations_executed': False,
              'numa_allocation_executed': False, 'tests_executed': False}
    write_json(run / 'readiness.json', status)
    return status


def infer(config, run, prompt):
    if config.profile != 'dram':
        raise ValueError('CXL arena profiles are not model KV offload; initial inference requires the dram control profile')
    model = config.values['model']
    if not model['device'].startswith('cuda:'):
        raise ValueError('initial 8B model launcher requires an explicit CUDA device; CPU trace/index tools are separate')
    ordinal = int(model['device'].split(':')[1])
    gpu = next((v for v in gpu_status()['devices'] if v['ordinal'] == ordinal), None)
    if gpu is None:
        raise RuntimeError('requested GPU is unavailable; no CPU fallback')
    if gpu['used_bytes'] > 128 * 1024**2 or gpu['utilization_percent']:
        raise RuntimeError('GPU is already in use; no model started. Coordinate a resource window before launching.')
    budget = config.values['resources']['hbm']
    if gpu['total_bytes'] - gpu['used_bytes'] < budget:
        raise RuntimeError('GPU free memory is below the configured job budget')
    if model['gpu_memory_fraction'] * gpu['total_bytes'] + 1024**3 > budget:
        raise ValueError('PyTorch memory fraction must leave at least 1GiB for allocations outside its allocator')
    args = [str(ROOT / 'scripts/with-env.sh'), '--cuda', 'cxl-kv-infer',
            '--model', model['path'], '--revision', model['revision'], '--local-files-only',
            '--dtype', model['dtype'], '--device', model['device'], '--backend', model['backend'],
            '--selector', model['selector'], '--layers', ','.join(map(str, model['layers'])),
            '--groups', ','.join(map(str, model['groups'])), '--max-input-tokens', str(model['max_input_tokens']),
            '--max-new-tokens', str(model['max_new_tokens']), '--gpu-memory-fraction', str(model['gpu_memory_fraction']),
            '--cpu-threads', str(config.values['workers']['torch_threads']), '--top-k', str(config.values['index']['selected_k']),
            '--prompt', prompt, '--output-dir', str(run / 'inference')]
    command = ['bash', *args]
    environment = {**os.environ, **config.child_environment()}
    write_json(run / 'launch.json', {'argv': command, 'gpu_admission_snapshot': gpu,
               'cpu_affinity': config.values['workers']['cpus'], 'environment_overrides': config.child_environment(),
               'scope': 'PyTorch allocator limit plus planned headroom; not a universal GPU allocation cap',
               'model_kv_placement': 'CUDA HBM; standalone arena configuration does not offload DynamicCache'})
    with worker_affinity(config.values['workers']['cpus'], config.values['tiers']['cpu_dram'],
                         allow_smt=config.values['workers']['allow_smt']):
        return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'infer'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
        sub.add_argument('--profile', default='dram')
        sub.add_argument('--output-dir', type=Path)
        if name == 'infer':
            sub.add_argument('--prompt', required=True)
    args = parser.parse_args()
    run = (args.output_dir or ROOT / 'runs' / (
        'infra-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])).resolve()
    try:
        run.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f'cannot create new run directory: {exc}', file=sys.stderr)
        return 2
    try:
        record_manifest(run)
        write_json(run / 'requested.json', {'command': args.command, 'profile': args.profile,
                   'config_path': str(args.config.resolve())})
        (run / 'requested-config.json').write_bytes(args.config.read_bytes())
        config = ExperimentConfig.load(args.config, args.profile)
        prepare(config, run)
        code = infer(config, run, args.prompt) if args.command == 'infer' else 0
        write_json(run / 'summary.json', {'status': 'prepared' if args.command == 'prepare' else
                   'completed' if code == 0 else 'failed', 'command': args.command, 'returncode': code,
                   'model_inference_executed': False if args.command == 'prepare' else None,
                   'model_evidence': 'inference/summary.json' if args.command == 'infer' else None,
                   'tests_executed': False})
        print(f'Artifacts: {run}')
        return code
    except Exception as exc:
        (run / 'exception.txt').write_text(traceback.format_exc())
        write_json(run / 'summary.json', {'status': 'failed', 'error': type(exc).__name__, 'message': str(exc),
                   'command': args.command, 'tests_executed': False})
        print(f'{exc}\nFailure preserved: {run}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
