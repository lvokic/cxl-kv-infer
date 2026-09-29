# CXL-KV Replay

固定逐 head 选集，探索 GPU / DRAM / CXL 上的 sparse attention 执行路径。

**起始点：单个 KV group 的 trace replay → 单层 → 模型 decode。** 先用 NumPy 参考实现和 FlashInfer 建立相同 mask 的执行基准；再加入 NUMA store、CPU partial 和精排数据复用。首轮研究对象是候选 K 精排之后，`K + V`、`exact logits + V`、CPU partial 三条路径的成本。

## 交付状态

2026-09-29：本仓库提供初始源码和交接文档。**尚未安装依赖、运行程序、编译 kernel、执行测试或测量硬件性能。** 源码检查时的当前 Python 没有 NumPy / PyTorch / FlashInfer。

| 已提供源码 | 边界 |
|---|---|
| Trace v1：NPZ、逐 head CSR、模型与位置元数据 | 一个 request / step / layer / KV group |
| FP64 selected attention 与 partial merge | 数学参考，不是优化 CPU executor |
| `make-synthetic` / `inspect` / `replay` CLI | 小规模接口探索；不输出性能数据 |
| FlashInfer paged decode adapter | 尚未运行；per-head 逻辑行，selected KV 先复制到 GPU |
| 硬件快照脚本 | 只读；不启动 benchmark、不调整 NUMA 策略 |

NUMA/CXL allocator、真实模型 capture、Faiss selector、KLease 生命周期、优化 CPU kernel、`logits + V` GPU kernel、计时框架和在线 planner 都是后续任务，见交付文档。NPZ load 不会把数据放到 CXL，平台 JSON 也不自动控制分配。

## 先读这些文档

1. [代码起始交付文档](docs/HANDOFF.md)：推荐起点、任务顺序、完成标准。
2. [数据与执行契约](docs/ARCHITECTURE.md)：mask、trace、partial、KLease、位置与精度。
3. [实验方案](docs/EXPERIMENTS.md)：对照组、计量边界、继续/停止条件。
4. [环境与源码入口](docs/ENVIRONMENT.md)：NUMA 放置、依赖准备、固定源码版本。
5. [研究依据](docs/PROVENANCE.md)：proposal、知识库和论文的来源及散列值。

## CPU 参考路径的使用方法

以下是准备后执行的命令，本次交付没有执行。使用独立虚拟环境：

```bash
cd ~/cxl-kv-replay
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/cxl-kv-replay make-synthetic --output data/synthetic/group.npz
.venv/bin/cxl-kv-replay inspect data/synthetic/group.npz
.venv/bin/cxl-kv-replay replay data/synthetic/group.npz --output runs/reference.npz
```

`make-synthetic` 创建四个 Q heads 共享一份 K/V 的小样本；选集是随机生成的，只用于接口探索。`replay` 默认执行 FP64 参考计算，保存 `output`、自然对数 `lse` 和可追溯摘要。所有输出文件采用独占创建，不会覆盖已有结果。

FlashInfer adapter 的命令在 [环境文档](docs/ENVIRONMENT.md) 中。GPU 依赖需要先确认本机 toolkit / PyTorch / FlashInfer 的可用组合；`pyproject.toml` 中 extras 只是依赖名称，没有构成经过验证的锁定环境。

## 第一个研究交付物

对同一份真实 trace，产出一张路径成本表：

```text
Q ready → candidate K read/refine → selected K/logits handoff
        → V read / H2D / attention → partial merge → output ready
```

固定模型精度、逐 head mask、资源预算和数据位置，报告各阶段时间及独立定义的字节数。先证明简单路径之间存在可利用的差异，再增加组内调度复杂度。

研究知识库仍位于相邻的 `~/kvcache-wiki`。本仓库保存实现与实验；论文原件和完整文献综述继续留在知识库。
