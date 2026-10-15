"""One recorded experiment configuration for model, workers, storage and index."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from index.backends import backend_spec
from module.config import InferenceConfig
from .resources import ResourceLedger
from .topology import Topology


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    profile: str
    values: dict

    @classmethod
    def load(cls, path, profile='dram'):
        path = Path(path).expanduser().resolve()
        values = json.loads(path.read_text())
        if set(values) != {'model', 'workers', 'resources', 'tiers', 'profiles', 'index', 'pools', 'benchmarks'}:
            raise ValueError('experiment configuration has missing or unknown top-level fields')
        if profile not in values['profiles']:
            raise ValueError(f'unknown profile {profile!r}')
        result = cls(path, profile, values)
        result.validate()
        return result

    @property
    def placement(self):
        return self.values['profiles'][self.profile]

    def validate(self):
        v = self.values
        cpu, cxl = v['tiers']['cpu_dram'], v['tiers']['cxl']
        if not cpu or not cxl or set(cpu) & set(cxl):
            raise ValueError('CPU/DRAM and CXL must be disjoint, explicit node pools')
        topology = Topology()
        topology.require_workers(v['workers']['cpus'], cpu, allow_smt=v['workers']['allow_smt'])
        if any(type(v['workers'][k]) is not int or not 0 < v['workers'][k] <= len(v['workers']['cpus'])
               for k in ('index_threads', 'torch_threads', 'blas_threads')):
            raise ValueError('library thread counts must fit the configured worker CPU set')
        if v['workers']['builder_concurrency'] != 1:
            raise ValueError('initial infra supports one admitted builder')
        tier = self.placement['backing_tier']
        if tier not in ('cpu_dram', 'cxl') or self.placement['policy'] != 'extent_round_robin':
            raise ValueError('initial storage requires a tier and extent_round_robin placement')
        nodes = self.placement['backing_nodes']
        if not set(nodes) <= set(v['tiers'][tier]):
            raise ValueError('backing nodes must belong to the selected logical tier')
        topology.require_nodes(nodes, memory_only=tier == 'cxl')
        ledger = self.make_ledger()
        for node in nodes:
            if ledger.limits.get(f'node:{node}', 0) <= 0:
                raise ValueError(f'no positive allocation budget for node {node}')
        for node in topology.online:
            if ledger.limits.get(f'node:{node}', 0) > topology.nodes[node].memory_bytes:
                raise ValueError(f'node {node} budget exceeds observed physical capacity')
        if v['pools']['replacement_groups'] != 1 or v['pools']['packet_slots'] != 2:
            raise ValueError('initial configuration admits one extra group version and two packet slots')
        if any(type(x) is not int or x <= 0 for x in v['pools'].values()):
            raise ValueError('pool limits must be positive integers')
        self.inference_config().validate()
        if not re.fullmatch('[0-9a-f]{40}', v['model']['revision']):
            raise ValueError('record a fixed canonical model commit')
        index = v['index']
        if index['metric'] != 'inner_product':
            raise ValueError('attention retrieval requires inner product, not implicit cosine/L2 conversion')
        for choice in (index, index['control']):
            spec = backend_spec(choice['backend'])
            if choice['kind'] not in spec['kinds'] or index['metric'] not in spec['metrics']:
                raise ValueError(f'{choice["backend"]} does not admit this index kind/metric')
        if index['dimension'] != 128 or not 0 < index['nprobe'] <= index['nlist']:
            raise ValueError('Llama-3.1-8B pilot requires dimension 128 and 0 < nprobe <= nlist')
        integers = ('nlist', 'nprobe', 'max_vectors', 'training_iterations',
                    'workspace_bytes', 'candidate_k', 'selected_k')
        if any(type(index[k]) is not int or index[k] <= 0 for k in integers):
            raise ValueError('index capacities and training/query parameters must be positive integers')
        if not 0 <= index['seed'] < 2**31 or not index['selected_k'] <= index['candidate_k'] <= index['max_vectors']:
            raise ValueError('record a valid seed and selected_k <= candidate_k <= max_vectors')
        if (any(choice['kind'] != 'flat' for choice in (index, index['control'])) and
                index['max_vectors'] < 39 * index['nlist']):
            raise ValueError('IVF capacity must support the explicit training-prefix requirement')
        if index['backend'] == 'hakes':
            options = index['backend_options']
            if (options.get('pq_bits') != 4 or type(options.get('pq_m')) is not int or
                    options['pq_m'] <= 0 or options['pq_m'] % 2 or
                    options.get('reduced_dimension') != index['dimension'] or
                    index['dimension'] % options['pq_m']):
                raise ValueError('HAKES pilot uses PQ4 with even M dividing d and no dimension reduction; '
                                 'training/bindings and reduced-query stride remain integration work')

    def make_ledger(self):
        return ResourceLedger(self.values['resources'], cpu_nodes=self.values['tiers']['cpu_dram'],
                              cxl_nodes=self.values['tiers']['cxl'])

    def inference_config(self):
        model = self.values['model']
        return InferenceConfig(model=str(Path(model['path']).expanduser()), revision=model['revision'],
                               dtype=model['dtype'], device=model['device'], backend=model['backend'],
                               selector=model['selector'], local_files_only=True,
                               top_k=self.values['index']['selected_k'],
                               max_input_tokens=model['max_input_tokens'], max_new_tokens=model['max_new_tokens'],
                               layers=tuple(model['layers']), groups=tuple(model['groups']),
                               gpu_memory_fraction=model['gpu_memory_fraction'],
                               cpu_threads=self.values['workers']['torch_threads'])

    def child_environment(self):
        workers = self.values['workers']
        return {'OMP_NUM_THREADS': str(workers['index_threads']),
                'OPENBLAS_NUM_THREADS': str(workers['blas_threads']),
                'MKL_NUM_THREADS': str(workers['blas_threads']),
                'NUMEXPR_NUM_THREADS': str(workers['blas_threads']),
                'OMP_DYNAMIC': 'FALSE', 'TOKENIZERS_PARALLELISM': 'false',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
