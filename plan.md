# 多轮 IVF 与 NUMA 分层 KV 系统探索计划

更新：2026-10-09。本文是实施与探索计划，任务均为待执行；本轮仅进行了只读系统探测和资料核对，没有安装依赖、运行测试、模型或性能 benchmark。

起步组合：**Transformers/PyTorch 模型接入 + Faiss CPU IVF 控制路径 + 自管索引生命周期 + FlashInfer 执行基线**。先完成 persistent session、单 KV group、单层的因果数据流，再研究约 1M 向量规模下的追加、复用、替换，以及统一 CPU/DRAM 层 `{0,1}` 与 CXL 层 `{2,3}` 的协同。GPU kernel 的投入由阶段测量决定。

设计依据：[当前 proposal](docs/PROPOSAL.md)、[用户提供的 proposal](docs/CXL_KV_Research_Proposal_2026-09-28.md)、[架构 HTML](docs/CXL_KV_Architecture_v2.html)、[架构契约](docs/ARCHITECTURE.md)、[索引生命周期](docs/INDEX_LIFECYCLE.md)。本计划细化执行顺序；不替代这些语义契约，也不把目标系统描述成已有实现。

## 1. 首轮要回答的问题

1. **IVF 是否适合真实 attention 小批查询？** 在逐 head 质量要求下，需要扫描多少 lists/K？查询、精排与插入能否比强 scan 基线更合算？
2. **何时复用、何时维护？** 固定 centroid、增量 compaction、重新训练和整体替换，在多轮 session 中各自何时能摊销？
3. **替换能否长期有界？** append log、exact tail、building/retired 版本和 scratch 是否保持上限？追赶、发布和回收是否产生尾延迟？
4. **CPU/DRAM 与 CXL 两层如何协作？** 用统一 CPU worker/DRAM 池管理索引、维护和 partial，将冷 KV 放入 CXL 池；在质量与容量约束下如何降低读取、维护和执行总成本？
5. **检索到执行是否浪费 K？** KLease / exact-logit handoff 能否减少重复读取、H2D 和 QK 计算？

优先回答 1–4。若固定参数、成熟 executor 已足够，接受收敛到简单设计，不以完成新 kernel 或复杂 planner 为目标。

## 2. 已核对的起点与资源边界

### 2.1 平台观察

2026-10-09 只读观察；原始快照见 [本轮平台快照](runs/platform-plan-20261009-readonly.json) 和 [前一轮快照](runs/platform-exploration-20261009-readonly.json)。占用会变化，每次运行重新核对。

| 资源 | 已观察配置 | 起步用途 |
|---|---|---|
| CPU | 2 × Intel Xeon 6530P；每 socket 32 物理核、64 逻辑 CPU | CPU 检索/精排、后台维护、NUMA worker 对照 |
| GPU | A100-PCIE-40GB，BDF `0000:8b:00.0`，SM80 | 模型、有限热 KV、selected attention |
| GPU 链路 | 16 GT/s ×16；PCI sysfs NUMA node 1 | 优先 node 1 的本地 staging |
| node 0 | 64078MiB；CPU `0-31,64-95` | 统一 CPU/DRAM 层的成员 |
| node 1 | 64436MiB；CPU `32-63,96-127` | 同一 CPU/DRAM 层；GPU-local 路径可作放置优化 |
| node 2 | owner 指定为 CXL；当前未出现在 online/has_memory，容量未观测 | CXL 层成员，当前不可用于实测 |
| node 3 | 262144MiB，无 CPU | CXL 层当前观察到的在线成员 |
| CPU 特性 | AVX2/AVX-512、BF16、AMX | 后续 CPU kernel 候选；指令支持不代表已测加速 |
| 系统策略 | `numa_balancing=1`；THP enabled/defrag 均为 `madvise`；`perf_event_paranoid=4` | 记录策略与实际驻留；硬件 counters 的可用性需另核实 |
| 存储 | 本地约 137GiB 剩余；NFS 约 62TiB 可用 | 本地临时运行；个人 NFS 保存模型/benchmark/归档 |

核对到 CPU 0/64 是同一物理核的 SMT siblings，CPU 32/96 同理。第一轮先使用每个物理核的一个逻辑 CPU；SMT 单独消融，不把 128 个逻辑 CPU 当成 128 个物理核。

前一轮采样 GPU 已使用约 37029MiB、利用率 35%；node 1 的 `free` 约 1092MiB。`free` 不等于可用实验预算，也不能将系统 `available` 全部认领。当前不启动压测，后续先与 owner 协调资源窗口与单 job 上限。

**待核实的平台问题：** PCIe 可见 `33:00.0`、`b4:00.0` 两个 CXL 设备，但没有 `/sys/bus/cxl/devices`。请向实验室运维确认 node 2 未暴露的原因，以及 CXL node 2/3 的物理地址范围、设备/decoder 映射和计数器访问方式；若与实验室配置说明不符，反馈运维。NUMA distance 20 不证明 node 0/1 到 node 3 的路径、带宽或延迟相同。若多个 CXL 设备被合并到一个 NUMA node，不能凭 node ID 宣称已控制具体 endpoint。

### 2.2 代码与环境

- 项目 HEAD：`60e69caa3d5aff645dae29eec938ab36df86ba64`，已有文档工作区变化；旧实验以各自 manifest/source snapshot 为准。
- 现有环境：Python 3.12.14、NumPy 2.5.3、PyTorch 2.14.0+cu130、Transformers 4.57.6、FlashInfer 0.7.0、Triton 3.8.0。Faiss 与 pybind11 尚未安装。
- 本地 CUDA 13.0 编译器在项目 `.venv` 内，通过 [with-env.sh](scripts/with-env.sh) 使用；不依赖系统 PATH 中的 `nvcc`。
- 已有 ModelAdapter / KVCacheLifecycle / TokenSelector / AttentionExecutor、真实 capture 与 standalone replay。
- 当前 generation 每次结束 request；KV lease 借用 DynamicCache tensor，`release(completion)` 没有实际事件等待。持续 session、稳定 backing、动态 IVF、NUMA allocator 和跨设备 partial 合作尚未实现。
- 既有 CPU 482 项、GPU 合计 644 项数值检查；SmolLM2 FP16 20/20、FP32 18/18，BF16 16/20，四项严格 logits 失败保留。它们不是任务质量、IVF 或 CXL 性能证据，详见 [RESULTS](docs/RESULTS.md)。

## 3. 框架与语言选择

| 组件 | 首选实现 | 为什么这样分工 | 实现落点 |
|---|---|---|---|
| 模型/session/token | Python + 现有 Transformers/PyTorch adapter | 保留真实 Q/K/V、template 与位置控制，缩短首个正确原型路径 | 扩展 `src/module/` |
| trace/配置/实验/质量 | Python + NumPy | 易审计、易复现；FP64 仅数学参考 | 扩展 artifacts 与 `scripts/` |
| index baseline | Faiss CPU Python API + C++ 内核 | 可控 IP、Flat/SQ/PQ 对照；先分清算法与 lifecycle | 后续 `src/index/` |
| KV arena/NUMA/leases | C++17 + pybind11 + libnuma | 显式 allocation、页驻留、owner、完成事件与并发边界 | 后续 `src/storage/` |
| manifest/slots/log | C++17，先短锁和单 writer | 先获得安全发布与回收，避免无锁协议成为首轮依赖 | `src/index/` native 部分 |
| K 精排/CPU partial | 先 NumPy 控制；性能阶段 C++/BLAS，再 SIMD | 参考正确性和优化性能分开；逐候选 Python 循环不进入性能路径 | 后续 `src/retrieval/`、`src/execution/` |
| GPU attention/merge | FlashInfer | 复用已集成成熟后端，适配内部自然对数 LSE | 现有 executors/backends |
| GPU 热点原型 | Triton；必要时 CUDA C++/CUB/CUTLASS | 先快速探索，再处理精细访存、stream 与融合 | 有实测依据后增加 native kernels |

Python/C++ 边界采用批量 descriptor/数组和显式 handle，native 长任务释放 GIL；记录所有 copy 与 conversion。默认不把 Torch/Faiss object 的内部裸指针当稳定 owner；共享 tensor/数组时必须保留来源引用与完成事件。CPU 工具继续不依赖 CUDA/Faiss import。

先不迁移到 vLLM/SGLang。全模型持续 session 和数据路径稳定后，只选一个 serving 接入；用实现侵入程度、extend/decode、KV ownership、维护事件集成来选。相关参考：[HF cache](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)、[vLLM prefix cache](https://docs.vllm.ai/en/latest/design/prefix_caching/)、[SGLang attention backend](https://docs.sglang.io/docs/advanced_features/attention_backend)。

新目录只在对应组件实际实现时增加，不创建空 scaffold。本轮不调整现有 `src/` 布局。

## 4. 统一 CPU/DRAM 层与 CXL 层

### 4.1 逻辑层次与物理节点

按 owner 的平台划分，系统使用以下逻辑资源池：

| 逻辑层 | NUMA nodes | 系统责任 |
|---|---|---|
| GPU HBM | GPU 设备 | 模型、有限 hot KV、packet 与 GPU partial/merge |
| CPU/DRAM 层 | `{0,1}` | 统一 CPU worker 池和 DRAM 预算；IVF 索引、精排、维护、CPU partial、目录与 staging |
| CXL 层 | `{2,3}` | memory-only 容量池；canonical 冷 K/V，必要时承载冷 codes/postings |

node 0/1 不承担固定的“后台/前台”架构角色。query、append、builder 和 CPU partial 从同一个 CPU 资源池取 workers；调度器按核数、队列优先级与带宽 credits 分配资源。node 2/3 不运行 CPU workers。

统一逻辑池不意味着物理访问成本相同。allocator 保留 page→node，worker 保留 CPU affinity，便于必要的局部性优化和性能归因；GPU-local staging 优先 node 1 是数据路径优化，不把 node 1 定义成唯一在线 CPU 节点。

**实际观测与配置必须分开：** 当前内核 `online=0-1,3`、`has_cpu=0-1`、`has_memory=0-1,3`，不存在 `node2` sysfs 目录。因此设计中的 CXL 池为 `{2,3}`，当前观察到的可用子集仅 `{3}`。请向实验室运维确认 node 2 未暴露是否属于配置差异及恢复方式；不由实验程序上线内存、修改 BIOS/kernel 或重映射设备。当前不能报告 node 2 容量、驻留或双 CXL 节点带宽。

完整 `{2,3}` 配置在 node 2 缺席时应给出明确错误。为推进单 CXL 节点实验，可另选显式命名的 `cxl3_only` 配置，记录 `requested={3}, available={3}`；它是独立实验配置，不能静默替代双节点配置。

### 4.2 两层 allocator 与对象放置

| 对象 | 初始逻辑层 | 分配与优化原则 |
|---|---|---|
| canonical 冷 K/V | CXL `{2,3}` | K/V 分离、稳定 ID、按 extents 分配；仅用已确认 online 的配置 |
| centroids/list 目录/manifest | CPU/DRAM `{0,1}` | 全层统一管理；每份状态有唯一 owner，少量不可变元数据复制需绑定 generation |
| codes/postings/active index | CPU/DRAM `{0,1}` | 先按 layer/group 分配，再按压力考虑 list 分片；不默认全放 node 1 |
| replacement slots/scratch | CPU/DRAM `{0,1}` | 池化复用，与 active/retired index 共用 DRAM 容量预算；不固定 node 0 |
| append/exact tail | HBM 或 CPU/DRAM | backing-ready 前保留可读唯一副本，tail 有界 |
| candidate K/CPU partial | CPU/DRAM `{0,1}` | 精排与执行共享 lease；按实际消费位置选 arena，避免无必要复制 |
| H2D packet ring | CPU/DRAM `{0,1}` | 有限 pinned pool；node 1 优先候选，并与其他 placement 实测比较 |
| 低热度 codes/postings | 后续可选 CXL | 仅在明确 DRAM 预算下消融，不与 canonical KV placement 同时改变 |

逻辑接口建议为 `allocate(tier, bytes, pool, placement_policy)`。`tier` 定义预算和角色；`placement_policy` 决定该层内的具体 nodes。返回实际 node 分布、owner、ready/completion 与 charged bytes。不让调用者通过普通 host allocation 的标签声称已经在 CXL。

第一版对新匿名 arena 设置 VMA policy，触页后查询驻留。CPU/DRAM 与 CXL 的各物理 node 分别设硬上限，再汇总为逻辑层预算：`B_CPU=B_0+B_1`，`B_CXL=B_2+B_3`。不可用节点不进入实际容量总和；共享服务器上的可用预算由资源协调决定，不等于物理容量之和。

物理 placement 起步为 group/extent 分片，不重复 canonical K/V。层内页级 interleave 是后续指定 arena 的对照；不对整个进程 interleave，不把 CPU/DRAM 与 CXL 混成默认 interleave pool。

### 4.3 统一 worker 池与维护协调

先从 `{0,1}` 选择已协调的物理核组成一个 worker 池，候选总数 4/8/16；首轮每核只使用一个 SMT thread。query、append、refine 和 CPU partial 优先，builder 使用受限 concurrency/credits，并保留有限维护进度。

builder 的第一版 concurrency=1。全局只准有限 group 的额外版本占用内存；有 retired reader 未结束时不能为了利用空闲核无限建新版本。CPU 空闲也不意味着 CXL/DRAM/互连带宽空闲。

GPU submission/staging 可在 GPU-local 核上执行。其他角色不写死 CPU 编号或 socket，具体 affinity 是运行配置。Faiss/OpenMP/BLAS/native pools 显式限制并行，避免多个库各自占满两路 CPU。

公平对照先固定相同总核数、逻辑层容量和维护 concurrency，比较紧凑 affinity、跨 socket 分散 affinity、数据 locality-aware assignment；之后再增加总核数与容量。固定 socket 分工只作为可选消融，不能成为系统架构前提。

### 4.4 围绕两层资源池的实验 profile

| Profile | 配置 | 研究目的 |
|---|---|---|
| T0：CPU/DRAM 控制 | workers 与 index 属 CPU `{0,1}`，canonical K/V 也在该层 | 获得 CPU/DRAM 容量允许范围内的完整路径基线 |
| T1：单 CXL 节点 | 与 T0 相同资源/selector，canonical K/V 显式在 node 3 | 当前可推进的 CXL 路径；记录单节点条件 |
| T2：双 CXL 节点 | canonical extents 分片到 `{2,3}`，其余条件固定 | node 2 online 且映射确认后，测容量与总带宽扩展 |
| T3：层内 placement | CPU `{0,1}` 的 group 分片/局部性/interleave；CXL `{2,3}` 的 extent 分片策略 | 同层内部 locality、负载偏斜与互连代价；分开改变两层策略 |
| T4：持续维护 | T1 或 T2，加 append 与 bounded replacement | query/maintenance 的 CPU、DRAM、CXL credits 与 P99 干扰 |
| T5：冷热分层 | 固定 DRAM 热区与 CXL 冷区，分开调 K/V/codes placement | CPU 层作为 cache/工作集时的收益和容量边界 |
| T6：attention 执行 | 固定逐 head mask；CPU partial、GPU packet、固定 hot/history；staging placement 单独消融 | K/logits 复用、gather/H2D、partial/merge 的完整成本 |

T0→T1 固定相同 workload 与可比的 CPU/DRAM 预算，单独改变 backing；T1→T2 分开报告等工作集的带宽比较和增大工作集的容量比较。T2 不具备条件时标为待执行，不能从 T1 外推其收益。

先按独立 layer/group 对 CPU 层索引分片，再条件性研究单 group/list 分片。CXL 层先按稳定 token extents 分布 K/V；具体 node 的负载均衡可改变新 extent 的归属，不默认搬迁已有 KV。若做迁移，单独跟踪 layout epoch、旧 lease、复制和峰值成本。

CPU/GPU partial 的语义分区与物理 NUMA node 解耦。每 head 选集按 token 分成不交贡献，union 仅合并物理读，归并使用 normalized output + natural-log LSE。

### 4.5 校准与证据

先测完整 CPU→CXL 路径，再决定层内 affinity 与 allocation：

1. CPU worker 位于 node 0/1，分别读写 DRAM node 0/1 和已 online 的 CXL node 2/3，测 scan、真实 ID gather、append。当前仅有 backing `{0,1,3}`，node 2 项明确缺测。
2. node 2 可用后，测单 CXL 节点与双 CXL 节点相同工作集、相同总核数下的并发读取与写入；拓扑/设备路径未确认前不预设带宽可相加。
3. 两路 CPU 同时访问 CXL、query 与 builder/append 同时运行，测总吞吐、query P99 与 backlog；以 credits 控制干扰，不靠固定 socket 隔离假设。
4. CPU/DRAM 层中的 gather→pinned staging→GPU H2D，测 staging locality 与额外跨 socket copy；GPU direct/mapped CXL 是独立可选 backend。
5. 从约 64MiB 诊断 buffer 起步，按协调的 job budget 扩大至超过相关 cache；记录 cold/warm、工作集与 residency，不使用现有任务的内存。

每段 arena 记录申请的逻辑层/nodes、实际 residency、policy、bytes 与失败。`mmap→mbind→触页→自进程 move_pages(nodes=NULL)` 是拟议控制路径，不能用 file-backed/NFS page cache 代替受控分配。需要 H2D 的小池在注册 pinned memory 前后核对状态，失败不静默 fallback。

保持现有全局 NUMA balancing/THP/DNS/proxy 设置。权限或 CXL 计数器不具备时，保留失败与限制，报告逻辑 payload/copy bytes，不能声称实测物理 CXL 流量。参考：[Linux NUMA policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)、[mbind](https://man7.org/linux/man-pages/man2/mbind.2.html)、[move_pages](https://man7.org/linux/man-pages/man2/move_pages.2.html)、[CUDA 13.0 host memory API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html)。


## 5. IVF 与相关系统的选型

### 5.1 实现与 baseline 的角色

| 对象 | 优先级/用途 | 接入边界 |
|---|---|---|
| Faiss CPU FlatIP/IVF-Flat | 首轮控制与索引载体 | IP、训练、add、scan；写操作外部同步。标准版本保存 FP32 vectors，并非 ID-only |
| 自管 ID-only IVF | P2 后半至 P3 的受控实现 | 复用 coarse training/assignment，lists 只存稳定 ID；扫描 canonical K，单独测随机 gather |
| Faiss IVF-SQ8/PQ | 压缩候选对照 | 编码/扫描产生候选，再 canonical K exact refine |
| HAKES-Index | CPU 并发读写强基线 | 只接 index library；记录压缩训练、原向量副本、locks 与 visibility |
| Quake | 动态维护强基线 | local split/merge、维护策略；论文当前实现串行 search/update/maintenance |
| cuVS IVF | GPU 构建/查询对照 | `extend` 不自动证明 concurrent extend/search；先 event/串行控制 |
| RetroInfer | 系统与分段追加基线 | 原生质量 track 与固定 mask/机制消融分开 |
| SIVF/ElasticIVF | append publication/slab 参考 | 审计版本 L2、MAX_K=32；修补并验证前不列为 attention IP 性能基线 |
| InSituANN | CPU/GPU 分工、构建参考 | 公开 PQ overlay 不含完整在线插入管线，不当作已具备实时 KV 插入的实现 |

Faiss 原地加锁 append 是朴素 baseline，不是目标并发系统。目标先用 immutable base + bounded exact tail，之后用不可变 delta chunks/受控发布；读取中的 Faiss 对象不被直接修改。长期 append 最终交给明确同步的自管结构或已验证 library。

**选择标准是满足相同质量、query P99、可见性和内存约束后的 session 成本 Pareto 前沿。** 不按最大批量 QPS 单独选择“最好 SOTA”。真正实时更新至少要求：新 KV 对 attention 及时可见；index 尚未 ready 时有精确覆盖；backlog/tail 有界；读者不会看到半初始化条目；退役可收敛。

官方依据：[Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)、[索引存储](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)、[HAKES](https://github.com/nusdbsystem/HAKES-Search)、[Quake §8.2](https://www.usenix.org/system/files/osdi25-mohoney.pdf)、[cuVS](https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-ivf-flat)、[SIVF 代码](https://github.com/hpdic/ElasticIVF)、[InSituANN 更新范围](https://github.com/mindtravel/InSituANN-OpenSource)。

### 5.2 初筛参数与工作负载

| 参数 | 起步范围 | 约束 |
|---|---|---|
| 每 layer/group 向量数 | 64K、256K、1,000,000 | 先单 group；不把各层 K 混进同一库 |
| Q batch | 真实 GQA ratio；补 1/4/8 控制点 | SmolLM2 ratio=3、d=64；3B ratio=3、d=128；8B ratio=4、d=128 |
| nlist | 256、1024、4096 | 当前可见训练样本足量才启用大 nlist，不从未来 K 训练 |
| nprobe | 4、16、64、128 | 先粗筛质量/扫描成本；必要时加入全扫描诊断 |
| selected k | 256、1024、4096 | 不超过合法历史数；mandatory policy 单独固定并计数 |
| 压缩候选倍数 | 1、2、4、8 | 固定最终 k，记录候选 K refine 成本 |
| 编码 | Flat→SQ8→PQ16/PQ32（8bit） | PQ 子空间数与维度整除；OPQ/FastScan 后置独立消融 |
| assignment/training | 当前 prefix；iterations 5/10/20 | 保存训练 ID、seed 与耗时；bootstrap 足量前 exact tail/显式等待 |
| append batch | 1、128、1024、8192 | 保留真实 turn/decode 因果序列；控制批量实验另标注 |
| delta/tail 上限 | 256、1024、4096 tokens 初筛 | 是运行上限，不覆盖需求时回压；prefill burst 分块提交 |
| compute workers | 1、4、8、16 | 与 NUMA profile 联合，但不首轮全组合 |

先固定 Flat、统一 CPU 池中的少量物理核、一个 k，筛 nlist/nprobe；再加入压缩、维护、层间/层内 placement。每阶段只扩一个主要设计轴。

### 5.3 RetroInfer 的具体参考点

本地 [RetroInfer 更新路径](/home/huangyirui/RetrievalAttention/cache_hub/retroinfer_cache.py:595) 已做 1024-token 分段聚类、metadata 扩展和 buffer 更新；[模型 init](/home/huangyirui/RetrievalAttention/model_hub/llama.py:157) 会重新创建 request cache。借鉴逐层 build/offload、scratch 共享、wave buffer 和 CPU/GPU 语言分工。

重点比较其 metadata/centroid 长期增长、更新突刺、重新分配、CUDA Graph 失效与多轮保留。不能把分段 append、scratch reuse 本身作为新增贡献。其共同 cluster 选择与 estimation zone 不等于我方逐 head exact selection；其 CUDA copy 有 d=128 假设，不能直接用于 SmolLM2 d=64。pinned host path 也不等于实际 CXL path。

环境独立：RetroInfer 上游依赖不同于当前 `.venv`。HAKES/Faiss forks、Quake/LibTorch、cuVS 分别固定环境和源码，在独立 run 中执行同一 workload；不混装互相覆盖，不把额外 RPC/进程边界成本混成 index 核心性能。

## 6. 数据流与索引生命周期实现顺序

1. **Session：** `begin_session → append_turn → generate_turn → end_session`。验证 token prefix，维护 committed/pending token、EOS/template、position；turn 结束不销毁历史。
2. **KV owner：** 每个空间 `(session, layer, group)` 稳定 row ID，K/V 分离 arena 与目录；KV layout epoch 和 index generation 分开。
3. **Append：** new KV ready 后取得有界 owner；backing-ready 和 index-ready 分别记录。先初始化 payload/ID/code，再发布；完整水位不跨 hole。
4. **Read：** 短锁内安全 pin manifest；按本层 query position 限制可见范围。ANN base/delta 加完整 exact tail，逐 head 去重。
5. **Refine：** candidate union 仅合并物理 K 读取；各 head 精确 QK/top-k，保留 selected K lease/真 logits；确定选集后再取 V。
6. **Replace：** builder 取得 build 水位，扫描 canonical K；新增写入进入有界 log。若换 codebook，用新参数重编码 catch-up 数据。
7. **Publish：** 在有界 commit gate 内追赶至明确水位，整体发布 codebook/base/delta/tail/watermarks/epoch；新读者 pin 新版本，旧读者保留原视图。
8. **Retire：** 等 CPU readers、GPU kernels、DMA、cache install、builder leases 完成后复用槽位，更新 generation 防 ABA。失败 build 保留原 active；失败代价仍计量。

起点单 writer、一个后台 builder、显式 handles。每层/group 的 active index 全部保留；全局 credits 限制额外 building/retired bytes，首轮只允许一个 group 的额外替换任务。scratch 跨层复用；不能默认每层有一份完整双缓冲。

## 7. 内存预算与流量口径

预算按 CPU/DRAM `{0,1}`、CXL `{2,3}` 和 HBM 三层管理，同时保存每个 online 物理 node 的明细；node 2 当前容量未知，不计入可用预算。包含所有 live/reserved 对象：

```text
canonical K/V + active index + building/retired index
+ delta/log/exact tail + candidate K + selected K leases
+ hot KV copies + builder scratch + temporary relocation copies
+ pinned staging/packet ring + partial states + library/model workspace
+ allocator slack/fragmentation
```

单 group pilot 的拟议池配置如下，作为后续配置项而非当前已分配资源；执行前仍须满足已协调的各 node/job 总预算：

| 配置项 | 首个候选值 | 超限行为 |
|---|---|---|
| packet slots × slot bytes | 2 × 32MiB，CPU/DRAM 层；node 1 为 locality 候选 | 等待消费完成；无 free slot 不继续 gather |
| candidate K pool | 64MiB | 分块 refine；保留各 head 正确 top-k，不截断候选冒充原语义 |
| builder scratch | 64MiB，CPU/DRAM 层；按 profile 选物理 node | 分块 assignment/encoding；library 额外 workspace 也须预登记 |
| exact tail | 1024 tokens/group | 同步推进或回压；不会只保留最近窗口 |
| append log | 4096 records/group，另限 bytes | builder 追赶、推迟维护或回压；引用的 KV lease 同时收费 |
| 额外版本 | 全局一个 group 的 building 或 retired 槽位 | 未退役完成不启动下一 replacement |
| FlashInfer workspace | 沿用当前 128MiB 起点，单列 HBM | 按 backend 实际需求核对，不为每层重复隐式分配 |

池容量从小规模 pilot 调整，不能以增大 buffer 掩盖可见性/backlog 问题。跨 socket 临时版本→最终版本的 copy 试验需要额外峰值 credits；默认单额外版本预算不允许同时留两份完整 replacement 副本。

以 32 层、8 KV heads、N=1,000,000、d=128 为例，以下均为公式估算，不是实测：

| 内容 | 估算 payload | 说明 |
|---|---|---|
| BF16 canonical K+V | 122.07GiB | 不含 HBM 副本和额外 metadata |
| BF16 K 的完整额外副本 | 61.04GiB | 仅适用于实际 BF16 copy，不能套到标准 CPU Faiss |
| CPU Faiss 完整 FP32 K 副本 | 122.07GiB | 另加 ID、lists、centroids；小规模对照后必须纳入预算 |
| 32B code + 4B ID | 8.58GiB | 自管紧凑 ID 假设；含全部层/group，未计额外对象 |
| 32B code + 8B ID | 9.54GiB | 对应 64-bit ID payload 假设 |
| 单 group 的 32B code + 4B ID | 34.33MiB | 额外版本粒度可先限制到 group |
| 单层 8 groups 的同类 payload | 274.66MiB | 不能默认全层同时 rebuild 免费 |

先用单 group 控制峰值，确认 CPU/DRAM 层总预算和各 node 余量后扩层；不能在某个物理 node 受压时无检查地分配全量 Flat。索引代码/ID、K gather、V gather、H2D copy、维护扫描与迁移分别计 bytes。没有硬件 counters 时使用 `logical_payload_bytes`、`copy_bytes`、清楚标注的估算，不报告 measured CXL traffic。

拟议回压顺序：停止可选预取 → 限制维护 admission/并发 → 同步追赶或暂停新 append → 明确失败退出。必须保留查询可读的唯一 KV 副本；不能以丢 tail 或提前 reclaim 降低内存。

## 8. Benchmark 与 baseline

### 8.1 三类 workload

| Track | 用途 | 起步内容 |
|---|---|---|
| 可控 vector/event stream | 1M 生命周期、skew、更新/查询并发、内存压力 | 稳定、topic switch/return、长 turn burst；标为 synthetic |
| 真实因果 Q/K/V replay | 隔离 IVF、维护、NUMA、execution | 3B 8K/32K，单 group→单层；固定相同输入/选集 |
| 模型多轮 free generation | 任务质量、误差累积、TTFT/ITL | RULER 派生分轮注入、LoCoMo 多轮；与官方终点 track 分开 |

新增 session event trace 保存 append chunks、Q records、token/position、layer/group、版本/水位、arrival/turn boundary 和事件顺序；不能每个 decode step 重存完整历史 K/V。保留现有 Trace v1 作为单 step 检查输入。capture/写盘成本单列，性能 replay 从明确 placement 的内存输入运行。

真实短 trace 重复到 1M 时标为 repeated trace，只回答容量/协议问题；不能代表真实 1M RoPE/QK 分布或质量。单 session 的 query 与 append 有因果依赖，通用 ANNS 的任意读写比例只作辅助压力实验。

### 8.2 模型与任务选择

| 对象 | 用途与进入条件 |
|---|---|
| 固定 revision SmolLM2-135M | session/接口/数值控制；8192 position limit，d=64 |
| Llama-3.2-3B-Instruct | 首个研究模型，d=128、GQA、128K 配置；先 8K/32K |
| Llama-3.1-8B-Instruct | 后续强质量参照；先核实下载、许可、配置与容量 |
| NVIDIA UltraLong 1M checkpoint | 后期候选；EOS/vocab/template/RoPE、checkpoint dtype 与长 prefill 先验证 |
| RULER | multi-needle、tracking、aggregation；先 4K/8K/32K，每配置约 20 条 pilot，再扩独立 seed/样本 |
| LoCoMo | 先官方终点文本 QA，再独立派生 streaming replay；仅 10 conversations，按对话聚类报告 |
| LongMemEval | 后续长记忆/知识更新/时间推理；官方远程 judge 与本地替代评分分开 |
| LongBench/InfiniteBench/BABILong | 形成主线后补充覆盖，不首轮同时调参 |

NFS 个人目录 `data/models` 当前为空，`data/benchmarks` 有 RULER 目录，完整性待核对。归档路径沿用 `/mnt/nfs/home/huangyirui/data/{models,benchmarks}`，下载大文件优先实验室 HF mirror/ModelScope，记录实际 revision/hash；模型本地运行。公开重打包必须独立记录来源和许可，不假定与 Meta 原仓字节一致。

官方终点 QA 与派生 streaming 追问独立运行：仅在证据已经出现时提问，新增追问会改变后续上下文。teacher-forced 原对话 replay 与 free generation 分支分别报告。原官方评分若使用远程 API，本地 judge 结果明确标为替代协议。

参考：[3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)、[8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)、[1M 候选](https://huggingface.co/nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct)、[RULER](https://github.com/NVIDIA/RULER)、[LoCoMo](https://github.com/snap-research/locomo)、[LongMemEval](https://github.com/xiaowu0162/LongMemEval)。

### 8.3 最小 baseline 矩阵

| 轴 | 必须比较 | 固定条件 |
|---|---|---|
| 选择/质量 | dense、exact top-k、IVF-Flat、SQ/PQ+exact refine | 模型、dtype、mandatory policy、逐 head budget |
| 索引/维护 | exact FlatIP；Faiss lock append；never rebuild；每 turn rebuild；固定阈值 compact/retrain；后台替换；再 HAKES/Quake | 因果 stream、质量、CPU cores、内存预算 |
| memory management | new allocation vs slot reuse；logical reset vs clear；统一 CPU 池的维护并发与 locality；逐 group vs 逐层 | 复制/旧版本/失败均计费 |
| attention execution | FlashInfer GPU packet；优化 CPU；固定 hot/history；再 logits+V | 冻结同一逐 head CSR 与 dtype |
| 层次与 NUMA | T0/T1 先比较；T2 待 node 2 online；之后 T3/T4/T5/T6 | 等核数、等容量先测；扩资源结果单列 |
| 系统 | RetroInfer 原生 quality/E2E；对齐语义的机制消融 | 记录原生近似机制、更新触发和 CUDA Graph 状态 |

我方 `retr-attn` 的 NumPy exhaustive 模拟不代表原版 RetrievalAttention。原版如需加入，单独锁历史版本。未运行上游 artifact 时只标“机制复现”，不报告击败原系统。

## 9. Kernel 的进入顺序

| 优先级 | 候选 | 具体节省 | 进入条件 |
|---|---|---|---|
| 先复用 | FlashInfer attention/merge、cuBLAS、Faiss scan/training | 获得成熟控制 | 确认 SM80、dtype、mask 与 LSE 契约 |
| 高 | CPU span gather + exact refine/top-k | 随机读放大、重复 K 读取、Python 开销 | 测得 gather/refine 主导；C++ 控制后再 SIMD |
| 高 | GPU logits+V stable softmax/PV | K 再上传、QK 再算 | 真实 logits identity 一致；净省完整路径时间 |
| 中 | masked GQA shared-KV tile | 多 head 重复 KV 加载/launch | 原有独立逻辑行的冗余确为热点 |
| 中 | 批量 assignment/encoding/append | turn burst/build 开销 | microbatch+tail 后仍有必要；不恶化 decode P99 |
| 后置 | fused IVF probe/scan/top-k | compressed scan 与中间 materialization | 排除 K gather/传输主导；避免重写全套 ANN GPU engine |

内部 partial 为 normalized output + natural-log LSE；适配 FlashInfer 外部 LSE 底。每个 head 的各路贡献不重不漏，物理 union 不替代 membership。padding 不进 softmax，post-RoPE K 不因 packet offset 再做位置变换。

每项先记录阶段占比与可省工作，计入新增 descriptor、copy、workspace、launch、join 与资源竞争；仅在完整 attention/session 净改善时保留。不预设 SIMT/Tensor Core、Triton/CUDA 的赢家。[FlashInfer attention](https://docs.flashinfer.ai/api/attention.html)、[state merge](https://docs.flashinfer.ai/api/cascade.html)。

## 10. 工作包与拟议验收

以下验收是在后续实施与验证时执行的条件，本轮不添加或运行代码测试。

| 工作包 | 依赖 | 具体交付 | 拟议进入/退出条件 |
|---|---|---|---|
| P0：资源与可复现起点 | 无 | topology、依赖/源码锁定、resource budget、NUMA 校准方案 | CXL 映射/residency 方法明确；无资源窗口则仅进行 CPU 文档/代码工作 |
| P1：persistent session | P0 环境 | session API、增量 turn、pending token、full 控制与因果 trace | 与重算 full 控制在登记容差内一致；历史保留、position/template 无漏重 |
| P2：单 group IVF | P1；可控 stream 可提前做控制原型 | FlatIP/IVF-Flat、ID-only、exact tail、质量曲线 | 满足逐 head visibility；真实小批在可接受质量下有成本空间 |
| P3：有界替换与层次协同 | P2 | log/catch-up/manifest/retire、CPU/CXL pools、T0/T1/T4；T2 条件具备后加入 | 无 future/半条目/提前回收；tail/log/版本有界；长期替换可收敛 |
| P4：partial 与执行热点 | P3；与质量预算协调 | K/logits handoff、CPU/GPU 控制、T3/T5/T6 条件探索 | 同 mask 不重不漏；完整路径归因；确定必要 native/kernel 工作 |
| P5：整层/全模型质量与 CXL | P2–P4 适用部分 | RULER/LoCoMo、RetroInfer、任务质量、实际 placement/E2E | dense/exact-topk 先满足质量；区分真实长 prefill 能力与索引规模 |
| P6：优化与 serving 扩展 | P4/P5 有净空间 | 必要 kernels；一个 serving 接入候选 | 同语义、同资源下的重复实验净收益；否则保留成熟路径 |

第一批实施任务：

- [ ] P0-1 固定 baseline checkout、依赖候选与构建配置，先不升级现有 `.venv`。
- [ ] P0-2 与 owner 协调每个 job 的 CPU、各 node 内存、HBM、pinned memory 和运行窗口。
- [ ] P0-3 向运维核对 node 2 缺席及 CXL 映射/counters；记录 configured={2,3}、observed={3}，实现前明确 residency 方法。
- [ ] P1-1 扩展 generation/cache 为持续 session，保存 committed/pending token、template hash 与 position。
- [ ] P1-2 定义 append/Q/lifecycle event trace，保留现有 Trace v1。
- [ ] P1-3 后续验证短上下文 full session 与重算控制，保留所有失败。
- [ ] P2-1 接入 FlatIP/IVF-Flat IP，外部写锁；先一个 layer/group。
- [ ] P2-2 实现完整 exact tail 与逐 head 去重，导出候选、精排、copy/timing 分项。
- [ ] P2-3 获得真实 3B 小批 Q/K trace，粗筛 nlist/nprobe；确认是否需要压缩。
- [ ] P3-1 建立 CPU/DRAM={0,1}、CXL={2,3} 的逻辑 allocator，K/V 独立 arena、稳定 ID、page residency、批量 lease；节点可用性显式校验。
- [ ] P3-2 实现有界日志、单额外版本、catch-up、整体 publish、真实 retire。
- [ ] P3-3 在统一 CPU 池、相同总核数/预算下比较 DRAM 与 CXL backing、维护 credits/并发；再作层内 locality 消融。
- [ ] P3-4 运行批准规模的 64K→256K→1M stream；无界 backlog 或资源超限即停止扩长。

工期在 P1/P2 实测开发与运行成本后估算；不能预先将 1M 模型、多个强 baseline 和 native kernels 压进同一个首轮里程碑。

## 11. 指标、决策门槛与第一轮图表

### 11.1 必须保存的指标

- **正确性：** committed/indexed/backed/visible 水位、ID coverage、hole、重复贡献、lease/event、generation 和失败恢复。
- **有效性：** exact top-k recall、离线 retained attention mass、output/LSE/logits 误差、逐 turn/free-generation 任务分数。
- **效率：** query/append/build/catch-up/publish/drain P50/P99；首轮/新轮 TTFT、ITL、TPOT、session wall time、回压与维护干扰。
- **内存：** 每个 tier/node 的 live/reserved/peak、scratch、candidate K、tail/log、active/building/retired、packet ring、副本和页驻留。
- **NUMA：** worker 与 data node、同/异 socket、访问/复制 bytes、并发干扰、线程/SMT 与 actual residency；可用时再加硬件计数器。

先在验证集登记数值容差与质量要求；task quality 可拟定相对 dense 的绝对下降不超过 1 个百分点，再根据任务和统计能力确认。20 条 pilot 不足以证明 1pp 非劣；扩样后按 session/conversation 配对报告置信区间。Recall 0.95/0.99 是诊断扫描点，不是 attention 质量保证。

构建、捕获、profiler、JIT/warmup 与 timed steady state 分开。CPU monotonic 与 CUDA events 记录各自时间域；不拼接不同时钟，不用无条件 max 假设完美重叠。固定 workload 多 seed/独立 runs，P99 样本不足时明确标注，不作稳定尾延迟结论。

### 11.2 收敛条件

- 如果 exact top-k 不满足质量，先调预算/层范围，不继续优化 ANN。
- 如果 IVF 在真实小批量下需要接近全扫描，保留 scan 控制并收敛索引方向。
- 如果固定 centroid/调 probe 已足够，停止复杂 adaptive rebuild。
- 如果维护不能抵消后续查询收益，选择复用/局部维护；未来 Q 只作 offline oracle。
- 如果层内分片/interleave 引入的 copy、join 或 CXL 争用更大，保留更简单的 placement，逻辑 CPU/CXL 分层不变。
- 如果扩 CPU cores、DRAM 或 CXL 节点后才变快，分别报告扩资源收益与同资源策略收益；不外推缺测 node 2 的结果。
- 如果 tail、日志或 retired 版本无界、提前回收或 residency 失实，不接受性能结论。
- 如果 kernel 只赢局部时间而端到端无净收益，停止自研。

### 11.3 第一轮应交付的图

1. **CPU/CXL 层次路径矩阵：** CPU workers 0/1 × DRAM backing 0/1、CXL backing 2/3 的 scan/gather/append；当前 node 2 标缺测；单/双 CXL 节点与 staging/H2D 对照。
2. **IVF 质量—成本曲线：** 真实小批 query latency、候选/scan bytes、recall/mass/output error，Flat/SQ/PQ 分开。
3. **多轮维护时间线：** 每 turn query/append、build/catch-up/publish/drain、tail/backlog、active/building/retired peak，new allocation/slot reuse 对照。
4. **固定选集执行分解：** refine、V fetch、pack/H2D、plan/run、merge/join；whole-group 与 K/logits handoff。
5. **任务与系统结果：** 逐 turn 质量、TTFT/ITL 分布与 session 总成本；synthetic/replay/真实模型分面，不混成一条性能曲线。

## 12. 版本与交付纪律

每个 run 使用新目录，保存 config、commit + dirty snapshot、实际依赖、编译器/flags/SM80、模型/data/评分 revisions 与 hashes、trace、seed、placement、资源预算、结果与失败。checkpoint、benchmark 和归档只写本人 NFS 目录，遵循 [共享 NFS 规则](/mnt/nfs/AGENTS.md)。计时前完成 NFS 读取与输入 staging，cold-start/加载时间另报。

2026-10-09 调研定位到下列上游版本；它们是检查依据，不表示已安装、编译或复现。正式实验优先选兼容 release/固定 commit，再记录构建结果。

| 项目 | 调研版本 |
|---|---|
| 本地 RetroInfer checkout | `03f912c6e917c380d9d90c5ec85bb0f161ba53ef` |
| 本地 MInference checkout | `29ef1974fcb5a1440ccf92d7d7996900230fddcf` |
| Faiss | `6692d7ab3748e95d155c49b7821dc9ef70a2611a` |
| HAKES | `9fbe8b99371c70d854618cff6de60e9986f4bd95` |
| Quake | `0651c88c5c71343191ef06ee4ed292cb4a01e1aa` |
| ElasticIVF | `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546` |
| InSituANN | `c2491ee638c69c7c8e5932a3b693b255a72f5458` |
| cuVS | `f751cadd8096afc36d17a85e10936270b09958e1` |

第一阶段的完成标志是：一个可解释、可追溯、预算有界的持续多轮单层/group 原型，以及决定 IVF/NUMA/维护方向的测量结果；并不要求同时完成新 serving engine、通用并发 ANN 或整套 GPU attention kernels。
