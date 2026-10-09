# 研究入口与执行顺序

更新：2026-10-09。主线是多轮 session 的持续 IVF 检索、索引复用/替换、partial attention 与有界内存管理；具体实施与 NUMA 探索安排见根目录 plan.md。

平台按 node 0/1 统一 CPU/DRAM 层、node 2/3 CXL 层组织，不预设 CPU socket 的前后台分工。当前 node 2 未出现在内核 online 节点中，计划分别记录目标配置与实际可测节点。

## 文档分工

- [plan.md](../plan.md)：2026-10-09 具体实施与探索计划，包含 NUMA 放置/worker profiles、任务清单、baseline 矩阵和拟议验收；执行安排从这里开始。
- [PROPOSAL.md](PROPOSAL.md)：背景、研究边界、问题/贡献、组件方案、实验、预期结果和计划的完整中文提案。
- [ARCHITECTURE.md](ARCHITECTURE.md)：图、源码与目标组件映射、session/KV/index/execution 契约；以此为实现依据。
- [INDEX_LIFECYCLE.md](INDEX_LIFECYCLE.md)：append、catch-up、manifest 发布、槽位环、reader retirement 和容量的详细协议。
- [RESULTS.md](RESULTS.md)：仅记录已执行结果；[PROVENANCE.md](PROVENANCE.md) 记录输入与版本。
- [DEVELOPMENT.md](DEVELOPMENT.md)：当前环境和可运行入口。

## 当前优先级

1. 扩展现有 request runner 为 persistent session，处理增量 chat prefix、pending emitted token、EOS 和单调 position。先以短上下文 full attention 对照隔离模型正确性。
2. 单层、单 KV group 的 IVF-Flat ID-only base + exact tail，对 FlatIP oracle；记录完整 token visibility，所有 ANN 候选用 canonical K exact refine。
3. 二槽替换与真实 lease/完成事件：有界日志、catch-up、整体发布、失败继续使用原索引、reader drain 后复用。用 1M 可控 stream 验证生命周期，明确不作模型质量结论。
4. 比较固定/自适应维护与逐层内存复用；先确定质量与维护摊销空间，再扩真实多轮 trace/任务。
5. 依据测得热点增加 native NUMA gather/CPU SIMD，以及必要 GPU kernel；完成实际 CXL allocation/驻留与流量计量后，才报告硬件收益。

对应提案 M1–M6。Quest/retr-attn adapters 保留为已有接入实验，暂缓调优；其本地 NumPy exhaustive IP 不作为动态索引性能基线。图索引、多 GPU/PP、跨 session prefix sharing 不进入当前核心开发。

## 统一成本与语义

同时计算整个 session 的构建、插入、query、审计、维护、传输、attention 和回收成本。Query P99 与 turn TTFT、allocator live/reserved/peak、active/building/retired index、pending tail、candidate K 和 staging 均需记录。

固定 mask 的执行研究与改变 retrieval/quality 的索引研究分别测量。逐 Q head 的 S_h 是语义，union 仅优化物理读；partial routes 不重不漏。K/logits reuse 只有 identity、precision、position/score transform 一致才成立；不以 ANN 分数代替 logits。

原 K/V 不随索引 compaction 移动；codes/postings 连续不意味着原 K gather 连续。CXL、DRAM 和 H2D 的逻辑 payload、复制量、硬件 counters 各自命名；普通 host allocation 不是 CXL allocation。

## 当前证据与停止条件

现有数值/小模型集成提供 framework 起点；没有持续 session、动态 IVF、任务质量或 CXL 性能结果。本次文档修订没有新增实验，既有失败和 run 目录保留。

质量先由 full / exact top-k / ANN 分层诊断；若固定 centroids 或只调 nprobe 已足够，不增加复杂 rebuild controller。若成熟 whole-group executor 已接近最优，不实现复杂 group 分流或新 CUDA kernel。若维护、读者回收、GQA mask/merge 或模型其它部分占主导，报告失败边界并收敛方案。
