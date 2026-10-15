"""Explicit backend selection and read-only dependency/source inventory.

Only the Faiss control adapter is written. A pinned research candidate cannot be
instantiated until its native adapter exists; selection never falls back to Faiss.
This module uses the standard library and does not load any ANN/CUDA library.
"""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'configs/index_backends.json'
ADAPTERS = {'faiss_cpu': ('index.faiss', 'FaissCPUIndex')}


def backend_spec(name):
    catalog = json.loads(CATALOG.read_text())
    if name not in catalog:
        raise ValueError(f'unknown index backend {name!r}; choose an explicit catalog entry')
    return catalog[name]


def backend_inventory(name):
    """Describe written code and present assets, without claiming runtime readiness."""
    spec = backend_spec(name)
    dependencies = {}
    for distribution, expected in spec['dependencies'].items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        dependencies[distribution] = {'expected': expected, 'installed': installed,
                                      'matches': installed == expected}
    archive = None
    if 'source_archive' in spec:
        path = ROOT / spec['source_archive']
        observed = None
        if path.is_file():
            with path.open('rb') as stream:
                checksum = hashlib.sha256()
                for block in iter(lambda: stream.read(1024**2), b''):
                    checksum.update(block)
            observed = checksum.hexdigest()
        archive = {'path': str(path), 'exists': path.is_file(), 'sha256': observed,
                   'matches': observed == spec['source_archive_sha256']}
    source_tree = None
    if 'vendored_source' in spec:
        path = ROOT / spec['vendored_source']
        marker = path / 'UPSTREAM.md'
        source_tree = {'path': str(path), 'exists': path.is_dir(),
                       'provenance_marker': marker.is_file()}
    written = name in ADAPTERS
    return {'backend': name, 'spec': spec, 'adapter_written': written,
            'adapter_dependencies_present': written and all(x['matches'] for x in dependencies.values()),
            'dependencies': dependencies, 'source_archive': archive, 'source_tree': source_tree,
            'scope': 'metadata/source inventory only; adapter import, train, append and search not attempted'}


def create_index(space, ledger, settings, *, threads, control=False):
    """Construct exactly the requested backend, or fail before substituting one.

    control=True explicitly requests settings['control']; it is never automatic.
    The session/lifecycle controller must retain this owner until readers retire.
    """
    choice = settings['control'] if control else settings
    name, kind = choice['backend'], choice['kind']
    spec = backend_spec(name)
    if kind not in spec['kinds'] or settings['metric'] not in spec['metrics']:
        raise ValueError(f'{name} does not admit the requested kind/metric')
    if name not in ADAPTERS:
        raise RuntimeError(f'{name}@{spec.get("revision", "unrecorded")} is a selected upstream candidate; '
                           'its project native adapter is not implemented. No Faiss fallback.')
    inventory = backend_inventory(name)
    if not inventory['adapter_dependencies_present']:
        raise RuntimeError(f'{name} needs its exact optional dependencies: {inventory["dependencies"]}')
    module_name, class_name = ADAPTERS[name]
    adapter = getattr(importlib.import_module(module_name), class_name)
    parameters = {key: settings[key] for key in ('dimension', 'nlist', 'nprobe', 'max_vectors',
                  'seed', 'training_iterations', 'workspace_bytes')}
    return adapter(space, ledger, kind=kind, threads=threads, **parameters)
