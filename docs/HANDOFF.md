# 交付状态

更新：2026-09-29。

已完成：仓库 `.venv` 与锁定依赖；trace / FP64 reference / FlashInfer replay；Transformers 模型生成入口；真实 SmolLM2 trace capture 和回放。

源码直接位于 `src/`：通用模块在根层，模型推理代码放在 `src/module/`，后端放在
`src/backends/`；不再使用 `src/cxl_kv_replay/` 包目录。

接口已整理为共享 `ModelAdapter` / `ModelStep`、`KVCacheLifecycle` / `KVCacheLease`、
`TokenSelector` 与 `AttentionExecutor`；Transformers DynamicCache、SDPA、FlashInfer 和 FP64
reference 已接入。重构后 FP16 真实模型端到端检查 20/20 通过，运行记录位于
`runs/flat-src-validation-fp16-20260929-01/`（本地忽略目录）。

验证边界：真实模型 FP16 20/20、FP32 18/18 检查通过；BF16 有 4 项整模型 logits 检查失败，记录未覆盖。合成数值结果和逐项运行目录见 [RESULTS.md](RESULTS.md)。没有完成任务质量评估、CXL allocator 或 CXL 性能实验。

下一步按 [RESEARCH.md](RESEARCH.md) 执行：先选研究模型与任务数据并评估 sparse/full 质量，再做固定选集的 DRAM/CXL 执行路径对照。

架构约束：[ARCHITECTURE.md](ARCHITECTURE.md)。环境与命令：[DEVELOPMENT.md](DEVELOPMENT.md)。代码来源：[PROVENANCE.md](PROVENANCE.md)。
