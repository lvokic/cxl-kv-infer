# CXL-KV Replay

面向持续多轮、超长上下文 KV 的 IVF 向量检索与 partial attention，探索 GPU / DRAM / CXL 上的更新、维护和执行成本。

**起始点：persistent session trace → 单个 KV group → 单层 → 模型 decode。** 先实现跨 turn KV 保留、IVF 可见性与 exact tail，再研究 compaction/rebuild 与 partial attention；随后扩到 NUMA DRAM 和 memory-only CXL。

## 交付状态

2026-10-15：技术选型、基础 infra 与 HAKES submodule 已更新，见 [开发与运行](docs/DEVELOPMENT.md)。
本地 Llama-3.1-8B-Instruct 为主模型；安装了锁定 Faiss CPU/pybind11，编译了 C++17/libnuma
arena 桥接层，执行了不加载模型的只读预检。新增统一配置、资源 reservation、逐 worker affinity、
独立 FlatIP/IVF-Flat API 与增量 event journal。尚未执行这些新接口的数值、驻留或性能验证。

IVF 选型已按并发更新需求调整：**当前优先采用 HAKES-Index 作为 CPU IVF-PQ 源码，Faiss
保留为串行控制，Quake 为动态维护对照**。HAKES 固定版本通过 submodule `third_party/hakes` 引入；独立
SIFT1M CPU 探索已构建并运行；本机的忽略型 run 记录位于 `runs/hakes-sift1m-cpu16-20261015T024344Z/RESULTS.md`。
项目自己的 HAKES adapter 和模型索引调用仍未接通，因此运行时不会假装已使用 HAKES，也不会自动
降级到 Faiss。具体接入边界见 [plan §5](plan.md#5-ivf-与相关系统的选型)。

2026-09-29：完整模型入口已就绪，合成与真实模型数值结果均有保留。SmolLM2-135M 的 FP16 检查 20/20、FP32 控制 18/18 通过；BF16 有 4 项严格 logits 对照失败。暂未进行任务质量评估或 CXL 性能 benchmark，见 [RESULTS.md](docs/RESULTS.md)。

| 已提供源码 | 边界 |
|---|---|
| Trace v1：NPZ、逐 head CSR、模型与位置元数据 | 一个 request / step / layer / KV group |
| FP64 selected attention 与 partial merge | 数学参考，不是优化 CPU executor |
| `make-synthetic` / `inspect` / `replay` CLI | 小规模接口探索；不输出性能数据 |
| FlashInfer paged decode adapter | 已通过合成 BF16/FP16 数值检查；per-head 逻辑行，selected KV 先复制到 GPU |
| `src/module/` / `cxl-kv-infer` | Transformers 4.57.6；SDPA / FlashInfer / FP64 可选；真实模型单 group、单层与完整生成已运行 |
| full / exact-topk 与真实 capture | 保留逐 head mask；真实 post-RoPE trace 可交给原有 replay；完整 KV 仍驻留模型设备 |
| 硬件快照脚本 | 已执行，只读；不启动 benchmark、不调整 NUMA 策略 |

基础 NUMA allocator 已提供源码和编译产物，Faiss 是串行 index control，尚未接入模型 selector。
统一配置选中 HAKES；项目 adapter 尚未实现时会明确拒绝实例化，不会自动替换成 Faiss。
持续 session、CXL KV offload、异步 KLease、index replacement、优化 CPU/GPU kernel 和在线 planner
仍为后续任务。NPZ load 不会把数据放到 CXL，普通 Transformers KV 仍在模型设备上。

基础 infra 的只读入口：

```bash
bash scripts/with-env.sh --cpu cxl-kv-infra prepare --config configs/experiment.json
```

每次生成新的 `runs/infra-*`。统一配置及 `dram` / `cxl3_only` / `cxl_dual` profiles 在
[experiment.json](configs/experiment.json)；配置限额不是已预约的资源，`prepare` 不分配这些池。

## 运行真实模型

当前 `.venv` 和模型权重已准备好：

```bash
bash scripts/with-env.sh cxl-kv-infer \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-files-only \
  --dtype float16 --backend flashinfer --selector exact-topk \
  --top-k 2 --sink-tokens 1 --window-tokens 1 --layers 0 --groups 0 \
  --prompt 'Explain why the sky is blue in one short sentence.' \
  --max-new-tokens 24 --capture --gpu-memory-fraction 0.1 \
  --output-dir runs/inference-first
```

每次使用新的输出目录。`--backend sdpa --selector full` 是 dense 控制组，`--groups all` 扩到整层。
`--model` 也接受本地权重目录。完整说明见 [DEVELOPMENT.md](docs/DEVELOPMENT.md)。

selector 可选 `full`、`exact-topk`、`quest` 和 `retr-attn`。Quest 用 `--top-k` 作为 token
budget，`--quest-chunk-size` 指定块大小；`retr-attn` 用 `--top-k` 作为检索数，
`--retrieval-sink-tokens` / `--retrieval-recent-tokens` 指定保留的首尾 token。示例：

```bash
bash scripts/with-env.sh cxl-kv-infer \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-files-only \
  --dtype float16 --backend sdpa --selector quest --top-k 1024 \
  --quest-chunk-size 16 --layers 0 --groups 0 \
  --prompt 'Summarize the passage.' --max-new-tokens 16 \
  --output-dir runs/quest-smoke
```

这两个 selector 是按 MInference 代码改接到本项目接口的检索/选集实现，不是直接调用其模型 patch。
`retr-attn` 当前用 CPU NumPy exhaustive inner-product 搜索模拟 MInference FlatIndex 的候选阶段；attention
executor 会对选中行重算精确 QK logits，selector 输出会记录 host K/Q copy bytes。它不代表 papyfaiss
索引性能或 RetrievalAttention 的端到端复现。

## 文档

- [项目介绍 HTML](docs/PROJECT_BRIEF.html)：背景与文献、动机对应设计、架构图、技术选型和预期效果；可离线阅读与演示。
- [具体探索与 NUMA 实施计划](plan.md)：任务依赖、框架/语言选型、统一 CPU/DRAM 层与 CXL 层协同、benchmark/baseline 与拟议验收。
- [交付状态](docs/HANDOFF.md)
- [中文研究提案](docs/PROPOSAL.md)：多轮 IVF、索引替换、效率/有效性/内存管理与实验设计。
- [架构与数据契约](docs/ARCHITECTURE.md)：模块分层、mask、trace、partial、KLease、位置与精度。
- [索引生命周期](docs/INDEX_LIFECYCLE.md)：追加、追赶、版本发布、槽位复用与安全回收。
- [开发与运行](docs/DEVELOPMENT.md)：环境重建、模型入口、trace CLI 和硬件起点。
- [研究计划](docs/RESEARCH.md)：当前优先级、实验顺序和计量口径。
- [已执行结果](docs/RESULTS.md)：合成与真实模型数值记录及其范围。
- [来源与版本](docs/PROVENANCE.md)：wiki 输入、上游代码 commit 和依赖版本。

## CPU 参考路径的使用方法

本机环境已经安装，可直接使用 `.venv/bin/cxl-kv-replay`。新 checkout 的锁定安装方法见
[开发文档](docs/DEVELOPMENT.md)。下列路径是使用示例；本次验证使用独立的
`runs/numerics-*/` 输入和输出，未创建这些示例路径：

```bash
cd ~/cxl-kv-replay
.venv/bin/cxl-kv-replay make-synthetic --output data/synthetic/group.npz
.venv/bin/cxl-kv-replay inspect data/synthetic/group.npz
.venv/bin/cxl-kv-replay replay data/synthetic/group.npz --output runs/reference.npz
```

`make-synthetic` 创建四个 Q heads 共享一份 K/V 的小样本；选集是随机生成的，只用于接口探索。`replay` 默认执行 FP64 参考计算，保存 `output`、自然对数 `lse` 和可追溯摘要。所有输出文件采用独占创建，不会覆盖已有结果。

FlashInfer adapter 使用 [开发文档](docs/DEVELOPMENT.md)中的 `scripts/with-env.sh` 启动，以选择仓库内 CUDA 13.0 编译器和链接路径。`requirements/` 保存安装版本和哈希；数值结论的范围见 [RESULTS.md](docs/RESULTS.md)。

研究优先级、成本表和停止条件见 [RESEARCH.md](docs/RESEARCH.md)。论文原件和完整文献综述保留在相邻的 `~/kvcache-wiki`。
