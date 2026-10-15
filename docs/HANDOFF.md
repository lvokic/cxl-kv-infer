# 交付状态

更新：2026-10-12。

2026-10-13 文档更正（没有新增执行结果）：owner 明确本轮已有大量历史/活动 IVF，轮内
只维护 delta 并定期按原参数入索引；全量重建仅在下一轮 prefill 开始启动。目标主路径是
GPU prefix/热 KV partial + CPU 历史 partial，经 LSE 合并。介绍页仍为 15 页，架构图恢复
为主体大图；实现依据同步至 [architecture](ARCHITECTURE.md)、[lifecycle](INDEX_LIFECYCLE.md)、
[proposal](PROPOSAL.md) 和 [plan](../plan.md)。现有 runner 尚未实现这些新路径。

本轮完成技术选型和基础 infra：主模型调整为本地 Llama-3.1-8B-Instruct，统一
`configs/experiment.json`；新增 `src/infra/`（配置、reservation、topology/worker affinity、
增量 journal、prepare/infer 入口）、`src/index/faiss.py`（串行 FlatIP/IVF-Flat 控制）、
`src/storage/`（C++17/pybind11/libnuma 严格 NUMA extent 和 Python tier adapter）。
没有新增空目录、`__init__.py` 或 `__main__.py`。

随后根据 owner 对并发 IVF 的要求更正选型：HAKES-Index 优先接入 CPU 并发路径，Faiss
只保留 exact/串行控制，Quake 用于动态维护对照。下载并静态核对了锁定 HAKES 源码；
新增通用 `index/interfaces.py`、显式 `backends.py` 和 `configs/index_backends.json`，
线程参数改为 `index_threads`，native/control 依赖锁分离。HAKES adapter 尚未实现，
所选后端不能自动降级为 Faiss；新的 prepare 会单列其未接入状态。

实际执行：在仓库 `.venv` 增加锁定 Faiss CPU 1.15.1 / pybind11 3.1.0，保留既有推理依赖；
NUMA bridge 编译成功，首次头文件冲突的失败日志保留；`cxl-kv-infra prepare` 只读预检完成。
记录见 `runs/native-build-20261012-infra-retry/`、`runs/infra-20261012-dram-readiness/`。
没有运行单元测试、数值检查、NUMA 分配、Faiss index 操作、8B 模型或 benchmark。
这些 prepare 记录对应原 infra 配置；后续选型修订未执行修改后的代码/prepare，
未构建 HAKES，也未运行并发、质量或性能验证。静态发现与接入步骤见 [plan §5](../plan.md#5-ivf-与相关系统的选型)。

范围：native arena 还未经过运行时驻留/生命周期验证；Faiss 内部 FP32 K 副本的容量 reservation
为预算估计，不是其实际 RSS；event journal 尚未接到模型持续 session。模型 KV 仍为 GPU
DynamicCache，没有 CXL offload、exact tail、二槽替换/retire 或跨设备异步 lease。

node 0/1 作为统一 CPU/DRAM 层，node 2/3 为 owner 指定 CXL 层；本轮仍仅观察到 0、1、3。
双 CXL profile 对缺失 node 2 明确报错，单 CXL 是显式独立 profile。待运维确认缺席节点和设备映射。

下一步是 P1：8B 的短上下文 dense 控制和 persistent session；通过登记的验收后进入单 group
HAKES filter/可见性/exact tail/原始 K 精排，同时保留 Faiss 控制。技术选择、运行命令和拟议验收见 [DEVELOPMENT.md](DEVELOPMENT.md)。

仓库已有独立 .venv、锁定依赖、trace / FP64 reference / FlashInfer replay、Transformers 模型生成入口及真实 SmolLM2 trace capture。源码位于 src/；模型接入在 src/module/，standalone 后端在 src/backends/。

共享接口为 ModelAdapter / ModelStep、KVCacheLifecycle / KVCacheLease、TokenSelector 与 AttentionExecutor。当前 DynamicCache 为 request 级，KV 仍在模型设备；借用 lease 没有 native 稳定 backing、实际异步 retirement 或 index generation。

已执行证据：CPU numerical run 482 项、GPU numerical run 总计 644 项；真实模型 FP16 20/20、FP32 18/18；BF16 16/20，四项严格 logits 对照失败。source/manifest/checks 与失败目录保留于 runs/；详见 [RESULTS.md](RESULTS.md)。这些不是 multi-turn 质量或 CXL 性能结果。

2026-09-30 交付新增中文 [研究提案](PROPOSAL.md)，重核 [architecture](ARCHITECTURE.md) 和 [index lifecycle](INDEX_LIFECYCLE.md)，并更新目标架构图；当时只修订文档，没有新增测试或 benchmark。本轮基础 infra 状态见本文开头。

下一步 M1：持续 session、incremental turn、pending token/EOS/template 与 full 控制；M2：单层/group 的 HAKES filter + exact tail、Faiss 控制及 Quake 维护对照；M3：有界二槽 build/catch-up/publish/retire。通过 correctness 与摊销分析后，再投入优化 kernel、实际 CXL placement 和全模型质量。详见 [plan.md](../plan.md)。

运行与环境见 [DEVELOPMENT.md](DEVELOPMENT.md)，输入/源码/图来源见 [PROVENANCE.md](PROVENANCE.md)。相邻 wiki 的 research proposal 与 architecture HTML 已按 owner 明确要求同步当前修订稿与英文图，旧稿由 wiki Git 历史追溯；论文材料和 kernel brief 保留。
