# 开发与运行

本仓库只使用自己的 `.venv`。环境和硬件配置是可追溯的起点；实测结果见
[RESULTS.md](RESULTS.md)，上游和研究输入版本见 [PROVENANCE.md](PROVENANCE.md)。

## 当前环境

| 部件 | 版本 / 配置 |
|---|---|
| Python / uv | 3.12.14 / 0.12.23（2026-10-12 观察） |
| NumPy | 2.5.3 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0，含 SM80 |
| FlashInfer | 0.7.0 (`cu13`) |
| Transformers / Hub | 4.57.6 / 0.36.2 |
| tokenizers / safetensors | 0.22.2 / 0.8.0 |
| CUDA compiler / JIT linker | nvcc 13.0.88 / nvjitlink 13.0.88 |
| CPU ANN 控制 / native binding | Faiss CPU 1.15.1 / pybind11 3.1.0；HAKES 仅下载源码，未构建 |
| Native 构建 / NUMA | CMake 3.28.3、G++ 13.3.0、libnuma 2.0.18 |

依赖和哈希分别保存在 `requirements/{core,gpu,inference}*.lock`；实际安装清单与原始安装
记录在 `runs/environment-20260929T054433Z/`。修改依赖时改 `.in` 并重新生成锁文件，
不要在共享环境中无记录升级。

当前 checkout 已就绪。如需重建完整推理环境：

```bash
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python --torch-backend cu130 --require-hashes \
  -r requirements/inference-cu130.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

仅 CPU trace 工具可以只安装 `requirements/core.lock`，再 editable install；
trace 包和 `cxl-kv-infer --help` 不导入 PyTorch / Transformers。CUDA 命令通过
`scripts/with-env.sh` 设置仓库 CUDA 13.0、SM80、JIT link alias 与本地编译缓存。
启动器不设置 GPU 可见性或 NUMA 放置。

## 技术选型与基础 infra

2026-10-12：正式主模型使用本地 Llama-3.1-8B-Instruct。SmolLM2 保留为既有数值控制，
3B 不再是必须经过的研究主模型；起点仍为单节点、单 session、单 layer/group。

| 组件 | 选择与实际落点 | 当前证据 / 下一步 |
|---|---|---|
| 模型 / session | Python、Transformers 4.57.6、PyTorch；`module/` | 旧单 request 入口保留；8B 尚未运行，persistent session 为 P1 |
| 配置 / artifacts / journal | Python/NumPy；`infra/` + 现有 artifacts | 统一配置与只读 prepare 已执行；增量 journal 源码未跑 |
| 主 index 候选 | HAKES-Index C++17 filter library、独立 pybind11 adapter | 锁定源码并静态核对；未构建、未实现 adapter、未运行；不导入标准 Faiss 代替 |
| index 接口 / 控制 | `index/interfaces.py`、显式 `backends.py`；Faiss CPU 控制 | FlatIP / IVF-Flat API 已写、控制依赖已安装；未运行 train/add/search |
| 稳定 host extent | C++17、pybind11、libnuma；`storage/` | 编译成功；尚未执行驻留或 owner/view 生命周期验证 |
| attention | SDPA dense 控制；FlashInfer selected attention | 沿用后端；新 8B 的数值/质量检查未进行 |
| 生命周期 | 自管 base/delta/tail、短锁、单 writer/单 builder | P2/P3；不在活跃 Faiss 对象上并发 search/add |
| kernel 优化 | 测量后才做 C++ SIMD、Triton 或 CUDA | 本轮无新 attention/ANN kernel；NUMA bridge 只是 OS allocation 边界 |
| 强基线 | RetroInfer 系统对照、Quake 维护对照、SIVF GPU 更新参考 | 各自固定 checkout/环境；不把 Quake 动态维护称为现成读写并发 |
| benchmark | 首选 RULER；后续 LoCoMo/LongMemEval | 当前 RULER 是 NeMo-Skills 入口源码，无生成数据；生成器环境分离 |

**HAKES 优先接入 CPU 并发 IVF；Faiss 仅为 exact/串行控制，Quake 为动态维护对照。**
这不表示已找到本机 attention workload 的性能最优解。HAKES 论文直接评估了并发读写，
Quake 论文 §8.2 的实现仍串行处理 search/update/maintenance；SIVF 处理 GPU 流式变更。
比较须对齐最终 selection 质量、因果 stream、query P99、更新可见延迟、线程和内存预算。
依据：[HAKES 论文](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)、
[Quake §8.2](https://www.usenix.org/system/files/osdi25-mohoney.pdf)、
[SIVF](https://arxiv.org/abs/2601.11808v3)。

目前标准 Faiss 的实际引用方式是 optional `faiss-cpu==1.15.1` wheel，在构造
`FaissCPUIndex` 时才 `importlib.import_module('faiss')`，使用 `IndexFlatIP` 或
`IndexIVFFlat(IndexFlatIP, d, nlist, METRIC_INNER_PRODUCT)`，调用 train/add_with_ids/search。
所有操作共用一个 `RLock`，包括 search/search；它只是可审计的串行控制。
官方 CPU 支持 concurrent search/search，但不保证 search/add 或 add/add，去掉 Python 锁
不能获得安全并发。标准 Faiss 保存额外 FP32 K，不能套用紧凑 code 的内存估算。
依据：[Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)、
[索引存储](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。

HAKES 也建立在修改版 Faiss 上，其 `faiss-ex` 增加了分区共享/独占锁和 PQ4 FastScan；
不能通过当前 PyPI wheel 获得这些机制。只接其 CPU filter library，使用独立命名的本项目
native binding，精排从 canonical K 读取；不安装其 RPC serving 或覆盖现有 Python Faiss。
filter-only `Initialize(..., mode=1)`、稳定 token ID、可见 prefix predicate、GIL 释放、
code/scratch 容量收费和版本 retirement 的具体实施顺序及静态发现见
[plan §5.2](../plan.md#52-hakes-接入顺序与已经发现的边界)。

### 安装与构建

native bridge 与 stock-Faiss control 的依赖锁已分开，均约束既有 inference lock。
基础 native 构建不需要 Faiss：

```bash
uv pip install --python .venv/bin/python --require-hashes -r requirements/native.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
.venv/bin/python scripts/build_native.py --output-dir runs/native-new --jobs 2
```

`runs/native-new` 必须为新目录。native 构建只用系统 CMake/G++/libnuma 和仓库 Python，
不依赖 CUDA，不修改系统库；保存源码、版本、binary hash、flags、日志与失败。
缺少 libnuma development headers 时明确失败，不改为普通 DRAM allocation。

仅在需要 Faiss 控制时安装 `requirements/ann.lock`；当前环境已安装，无需重复操作：

```bash
uv pip install --python .venv/bin/python --require-hashes -r requirements/ann.lock
```

HAKES 是源码构建候选，不包含在这两个 Python lock 中。其原始源包已经下载到
`artifacts/upstream/hakes-9fbe8b9.tar.gz`；revision/hash 见 `configs/index_backends.json`。
未构建 native library，未生成训练参数、未启动服务或 index 操作。

目录增加实际组件：`src/infra/` 为配置、拓扑、reservation、journal 和实验入口；
`src/index/` 为通用 identity/candidate 协议、显式 backend 选择/inventory 与 Faiss 控制；
`src/storage/` 为 native extent 和 tier allocator。
没有增加 `__init__.py`、`__main__.py` 或空目录。

`with-env.sh --cpu` 无需检查 CUDA toolkit；`--cuda` 设置已有 CUDA/SM80 环境，不带模式
参数时保持旧 CUDA 行为。CPU trace/reference 不要求 Faiss/PyTorch/native import；
Faiss 与 native bridge 分别在实际使用时才导入。

### 配置与入口

[experiment.json](../configs/experiment.json) 统一保存本地 model/revision、精度、`index_threads`、
逻辑层、物理 node 预算、index、pool 和 benchmark 路径；后端来源/能力声明独立记录于
[index_backends.json](../configs/index_backends.json)。初始值是小规模 pilot 的建议，不是已预约
的服务器额度，也不默认分配这些池：

- CPU/DRAM `{0,1}` 为一个池，示例 workers `0,1,32,33` 每物理核仅一个 SMT thread。
- 受管 CPU/DRAM reservations 共 1GiB，node 0/1 各 512MiB；CXL node 3 为 256MiB。
- 两个 packet slots 共 64MiB，candidate K / builder scratch 各 64MiB；tail/log/version
  上限已登记，实际池、回压与 replacement 控制仍属 P1–P3。
- HBM 计划上限 32GiB；PyTorch fraction=0.75，在 40GiB 卡上约 30GiB，另预留至少 1GiB。
  这不是驱动/FlashInfer 等所有 GPU allocation 的通用硬上限。
- 主 index 配置是 `hakes/ivf_pq_fastscan`，control 是 `faiss_cpu/ivf_flat`，以 IP/d=128、
  PQ4/M=32、无降维起步；候选倍数与质量需后续选取。这是选型配置，不表示 HAKES adapter 已实现。

只读准备，自动生成唯一新 run，不加载模型、不操作 index、不分配 arena：

```bash
bash scripts/with-env.sh --cpu cxl-kv-infra prepare --config configs/experiment.json --profile dram
bash scripts/with-env.sh --cpu cxl-kv-infra prepare --config configs/experiment.json --profile cxl3_only
```

`dram` 请求 backing `{0,1}`；`cxl3_only` 显式请求 `{3}`；`cxl_dual` 请求 `{2,3}`，
当前 node 2 缺席会拒绝，不静默替换为单节点。CXL 角色来自 owner，endpoint/decoder 映射
仍需运维确认。preflight 成功不证明 mbind/move_pages 权限或实际 allocation residency。

run 保存 requested/config、manifest/source.zip、topology/GPU 占用、model/download 来源、
baseline commits、benchmark 状态、依赖/native binary。核对模型当前大小与 metadata hash；
17GB 权重哈希使用既有完成记录，不在每次预检重算。

`index-backends.json` 与 readiness 分别报告 selected/control 的 `adapter_written`、
optional dependencies 和上游源包 hash。所选 HAKES 当前 `adapter_written=false`；
prepare 可以记录这一未完成状态，`prepared` 不表示 index 可运行。Faiss 不是基础预检的强制依赖。
构造目标后端会明确拒绝尚未接入的 HAKES；只有 `create_index(..., control=True)` 才显式
构造 Faiss control，不会自动降级。dense/full 模型控制还没有使用任何 IVF adapter。

显式启动 8B dense 控制的入口如下，本轮未运行这条命令：

```bash
bash scripts/with-env.sh --cpu cxl-kv-infra infer --config configs/experiment.json \
  --profile dram --prompt 'Explain the role of a KV cache in two sentences.'
```

launch 前重新检查 GPU 占用/预算，已有 GPU 作业时拒绝；CPU affinity 和各库线程限额用于
子进程。结果保存在新 run 的 `inference/`。这仍是单 request 和完整 GPU DynamicCache。
CXL arena profile 不等于模型 offload，infer 暂只接受 dram 控制 profile；正式资源窗口仍需协调。

### API 边界与拟议验收

`ResourceLedger` 在分配前申请 reservation，node charge 自动叠加其 tier，超限拒绝。
同一 run 的 index、allocator 和 buffers 应共享唯一 ledger，不为每个组件新建一套额度。
计数不包括未登记第三方分配，也不代表 RSS/物理 CXL traffic。Faiss 的容量/workspace
reservation 是估计，内部 allocator peaks 未可见；adapter 另收费 append ID 临时数组及
candidate 输出，后者由数组生命周期保留 charge。conversion、训练输入和 refine buffers
仍须调用者另行收费；调用者自行复制的候选也须另记账。

`TierAllocator.allocate(tier, bytes, purpose=..., nodes=...)` 返回 native owner，K/V 用
`allocate_kv` 独立分配。每个 extent 绑定单 node、按 extent round-robin；不对进程 interleave。
native 路径为 `mmap → MPOL_BIND → base-page first-touch → 全页 move_pages`；权限/驻留
错误会撤销本次分配。NumPy `region.bytes()` view 保留 owner/reservation；不提前 close，也不
自动注册 CUDA pinned memory。异步消费者必须持有 owner 到真实 completion；event retirement
尚未实现。依据：[Linux NUMA policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)、
[move_pages](https://man7.org/linux/man-pages/man2/move_pages.2.html)。

`FaissCPUIndex` 绑定 session/layer/group/KV epoch/transform identity；train 使用当前 prefix，
append 延伸连续 IDs，search 保留逐 head 候选。输入须为 contiguous FP32，不偷偷 conversion。
IVF 要至少 `39*nlist` 可见训练样本；不足时由下一阶段 exact tail 覆盖，不自动改 nlist。
不能查询早于训练水位的历史 prefix；retrain 需新 owner。mutation 失败后实例不可用。
候选分数不进入 attention logits，须由 canonical K 精排。

公共 `IndexSpace` / `CandidateBatch` / `CandidateIndex` 位于 `index/interfaces.py`，没有
Faiss 专有类型。后端须明确暴露其同步与 prefix 能力；通用协议本身不实现并发、snapshot
或精确 tail。HAKES 接入时扫描前过滤不可见 IDs，并将 generation lease 与完整 manifest
留给 controller。其外部同步要求、filter-only Reserve 和 query 变换边界须先处理。

`SessionJournal` 单 writer 保存 append/Q/lifecycle/stage，分别存新 K/V/positions 和 Q，不逐
decode step 重存历史；记录 origin/model/transform/dtype、space、turn/step、position/prefix、
generation 和 CPU monotonic 时间。它不使用 pickle、不覆盖目录，但尚未接入真实持续 session。
capture 磁盘 I/O 必须单列，不能作为 index 性能结果。

基础 infra 轮执行了安装、editable 注册、native 编译和 DRAM profile 只读 prepare。
首次编译失败在 `runs/native-build-20261012-infra/`，修复后构建在
`runs/native-build-20261012-infra-retry/`，预检在 `runs/infra-20261012-dram-readiness/`。
没有运行新 API 的单元、数值、驻留、生命周期、质量或性能测试，也没有运行 8B 模型。
随后的 IVF 选型修订仅下载/核对上游源码、修改代码/配置/文档和重新解析两个 lock；
没有执行修改后的 adapter/config/prepare，没有编译 HAKES 或运行任何测试/benchmark。

后续获得实施验证授权时的验收条件，本轮未执行：

1. budget 超限、失败回滚和 owner/views 生命周期；所有存活副本/缓冲仍计费。
2. node 缺席/权限失败/驻留不符必须拒绝，DRAM/CXL 分开记录实际页驻留。
3. FlatIP 控制、IVF candidates、稳定 IDs、失败 mutation、future-trained/旧 prefix 拒绝，
   并核对 FP32 副本、训练、candidate/workspace 峰值。
4. journal 的 append/Q/turn 顺序和 prefix 可重放，精度与来源一致，无重复历史 K/V。
5. 8B 短上下文 dense 与 persistent session 对照，核对 pending token/EOS/template/position，
   通过数值要求后再做质量与性能。

RULER 当前 commit `e8bbff677ca2c239640dc90f93310dcf32408c93` 的 README 指向 NeMo-Skills。
后续在隔离的生成器环境锁源码/模板/tokenizer/seed，导出 JSONL 后交给本项目 adapter，
不把其 serving engine 或依赖混装当前 `.venv`。生成数据、官方终点和派生 multi-turn track 未运行。

## 运行模型

模型模块在 [`src/module/`](../src/module/)。默认环境中已下载
HuggingFaceTB/SmolLM2-135M-Instruct，固定 revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`。权重缓存在忽略的 `data/models/`，run 输出在忽略的
`runs/`。模型/环境结果不能因此视为 CXL placement、质量评估或性能 benchmark。

```bash
bash scripts/with-env.sh cxl-kv-infer \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-files-only \
  --dtype float16 --backend flashinfer --selector exact-topk \
  --top-k 2 --sink-tokens 1 --window-tokens 1 --layers 0 --groups 0 \
  --prompt 'Explain why the sky is blue in one short sentence.' \
  --max-new-tokens 24 --capture --gpu-memory-fraction 0.1 \
  --output-dir runs/inference-new
```

已有输出目录会报错。`--model` 也接受含 config/tokenizer/safetensors 的本地目录；
`--prompts-jsonl configs/inference-prompts.jsonl` 用于顺序运行多个 prompts，每次请求有新的 cache。
默认是 BF16；目前跨后端严格整模型检查通过的是 FP16。CPU 控制运行显式指定
`--device cpu --dtype float32 --backend sdpa`。CUDA 不可用时会失败，不会悄悄改走 CPU。

`--backend sdpa --selector full` 是 dense 对照；`--groups all` 将目标 layer 的替换扩为所有
KV groups。`--layers all --groups all` 可把所有层交给该后端，但全模型 sparse quality 尚未
验证。capture 默认发生在第一个 decode forward，产物包含 post-RoPE Q/K、V、逐 head CSR、
模型 revision、输入/源码 hash 和 attention output；早停可能使目标 capture step 未执行。

选择器包括 `full`、`exact-topk`、`quest` 和 `retr-attn`。`quest` 的 `--top-k` 是 token budget，
并由 `--quest-chunk-size` 控制 chunk 大小；它按 MInference Quest 的 chunk upper bound 选块。
`retr-attn` 的 `--top-k` 是历史候选数，首尾驻留预算由 `--retrieval-sink-tokens`（默认 128）和
`--retrieval-recent-tokens`（默认 512）控制。当前实现以 CPU NumPy 精确 Flat-IP 搜索生成候选，
并在后端按原始 K 重算 attention logits；不是上游 `papyfaiss` 或其异步 cache/FlashAttention
执行路径。Quest chunk summary 当前每次 decode 重建；检索 selection 的 host K/Q copy bytes
会随运行记录保存。

## 运行 trace 工具

```bash
.venv/bin/cxl-kv-replay --help
.venv/bin/cxl-kv-replay make-synthetic --output data/synthetic/group.npz
.venv/bin/cxl-kv-replay inspect data/synthetic/group.npz
.venv/bin/cxl-kv-replay replay data/synthetic/group.npz --output runs/reference.npz
```

默认 replay 是 FP64 数学参考。synthetic 输入用于接口和数值对照；不会将 KV 分配到 CXL。

## 平台起点

硬件静态记录在 [`configs/platform.json`](../configs/platform.json)，用下面的脚本重新读取拓扑：

```bash
bash scripts/with-env.sh python scripts/inspect_platform.py --output runs/platform-new.json
```

已观察到 A100 PCIe 40 GB 位于 NUMA node 1，node 3 由 owner 标记为 memory-only CXL。
平台快照和静态配置都不能证明某个 KV allocation 实际驻留在那里。实验要对每种 allocation
单独 pin/first-touch/query residency，不要将整个进程绑定到 node 3。
