# 交付状态

更新：2026-09-30。

仓库已有独立 .venv、锁定依赖、trace / FP64 reference / FlashInfer replay、Transformers 模型生成入口及真实 SmolLM2 trace capture。源码位于 src/；模型接入在 src/module/，standalone 后端在 src/backends/。

共享接口为 ModelAdapter / ModelStep、KVCacheLifecycle / KVCacheLease、TokenSelector 与 AttentionExecutor。当前 DynamicCache 为 request 级，KV 仍在模型设备；借用 lease 没有 native 稳定 backing、实际异步 retirement 或 index generation。

已执行证据：CPU numerical run 482 项、GPU numerical run 总计 644 项；真实模型 FP16 20/20、FP32 18/18；BF16 16/20，四项严格 logits 对照失败。source/manifest/checks 与失败目录保留于 runs/；详见 [RESULTS.md](RESULTS.md)。这些不是 multi-turn 质量或 CXL 性能结果。

本次交付新增中文 [研究提案](PROPOSAL.md)，重核 [architecture](ARCHITECTURE.md) 和 [index lifecycle](INDEX_LIFECYCLE.md)，并更新论文风格目标架构图。文档明确高效率、有效性与内存预算，所有计划组件均标为待实现；没有新增测试或 benchmark。

下一步 M1：持续 session、incremental turn、pending token/EOS/template 与 full 控制；M2：单层/group 的 IVF-Flat + exact tail；M3：有界二槽 build/catch-up/publish/retire。通过 correctness 与摊销分析后，再投入优化 kernel、实际 CXL placement 和全模型质量。详见 [RESEARCH.md](RESEARCH.md)。

运行与环境见 [DEVELOPMENT.md](DEVELOPMENT.md)，输入/源码/图来源见 [PROVENANCE.md](PROVENANCE.md)。相邻 wiki 的 research proposal 与 architecture HTML 已按 owner 明确要求同步当前修订稿与英文图，旧稿由 wiki Git 历史追溯；论文材料和 kernel brief 保留。
