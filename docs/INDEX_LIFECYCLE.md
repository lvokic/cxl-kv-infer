# 多轮 IVF 索引生命周期：低成本追加、替换与内存复用

更新：2026-10-13。本文是目标协议与研究假设；持续 session、IVF 接入和版本槽位尚未实现。
基础 native extent 已编译，运行时驻留与回收尚未验证。研究叙事见 [PROPOSAL.md](PROPOSAL.md)，
组件契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 核心想法与适用范围

在一个持续多轮 session 中，prefill 分批产生新 K/V，decode 逐 token 产生新 K/V。累计 context 的本阶段目标上限约为 1,000,000 tokens；它是研究 workload 的规模，不是所有模型的最大能力。历史 KV 不随 turn 结束清空，向量索引持续追加、复用和维护。

本轮可以从已经积累约 100K tokens 的历史 KV 和活动索引 A 开始。轮内产生的新 K 进入有界 delta，查询同时覆盖 A 和完整 delta；delta 定期按 A 的既有参数批量插入 A。**轮内 delta flush 是增量插入，不是重训或全量重建；重建只能在下一轮 prefill 开始时获准并启动。** delta 大小、时间间隔、skew 等轮内统计可以决定插入批量或登记下一轮的重建需求，不能启动本轮 decode 中的全量 rebuild。

下一轮 prefill 开始时，利用已保留的历史 K 在另一槽位建 B，并接收本次 prefill 逐层产生的新 K；历史 KV 不为索引重建重新做模型前向。初版把完成追赶和发布作为该层/group 进入 decode 的条件，赶不上的工作计入本轮 prefill 延长与 TTFT；不以后台重建跨入 decode 掩盖成本。若开始时没有预算/空槽，则跳过本次重建，继续使用 A + delta。环形槽位管理分配与版本生命周期；发布开销可以与索引大小无关，构建、追赶写入、等待读者和清理的总成本仍需计量。

这属于 snapshot/delta、版本发布与内存池机制在 attention workload 上的设计适配，不能仅凭“双缓冲/指针切换”主张新颖性。[HAKES](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)、[SIVF](https://arxiv.org/abs/2601.11808) 与 [InSituANN](https://arxiv.org/html/2608.08751v1) 分别提供并发更新、发布安全和替换式维护的参照；wiki 的 [并发 IVF 分析](../../kvcache-wiki/wiki/analysis/concurrent-ivf-read-write.md) 区分它们的硬件边界。

## 2. Owner、地址与版本

每个 `(session, layer, KV group)` 是独立索引空间。GQA 的 Q heads 共享这份 K 索引，但各自保留候选和最终选集。token row ID 在 session 内稳定，与 token position、posting offset、物理地址、packet offset 分开。

canonical K/V 使用分离的 append-stable backing，通过目录映射 row ID 到地址。索引替换只改变 centroids、codes、postings、目录描述和索引 generation，不搬动 canonical K/V。HBM 热区是有预算的 resident 副本；刚生成且尚未下放的尾部 K/V 有单独 owner。

三个进度不能混为一个 epoch：

| 状态 | 含义 |
|---|---|
| KV 布局版本 | backing 目录及地址租约所绑定的布局 |
| 索引 generation | centroid/codebook、base 和 delta 描述的版本 |
| 可见 token 水位 | 本次查询可读取的完整前缀上界；另记录已 indexed/backed 的前缀 |

每层进度独立：layer 0 已生成当前 token 时，后续层可能还没生成它。查询只 pin 本层/group 的一致视图，并以自己的 query position 限制 causal 可见性。指针获取和 pin 必须受短锁或经过验证的安全读者协议保护；裸指针读取后再加 refcount 存在回收竞态。MVP 先用短锁、单 writer 和 refcount，不把无锁实现设为前置目标。

## 3. 追加与未入索引尾部

主路径为活动 IVF + 有界 delta，delta 中尚未发布到 IVF 的合法 K 必须精确扫描。
主接入候选为 HAKES 的并发 filter：本轮同 generation 的 routing/encoding 参数保持不变，
lists 可以在分区同步下追加。不可变 base + 独立增量段保留为控制变体，不能把 segment merge
和换参数的全量 rebuild 混为一次操作。
这仍是目标适配路径，公开库不自带本项目的 prefix snapshot、bounded tail 或 version retirement。
首批数据不足以训练时先走 exact tail，训练只消费当前已可用 K；learned-query 参数如需训练，
只消费已经出现的 Q，不能借用未来 decode Q。

每次查询 pin 完整 manifest 与可见水位，在分区读锁保护的一致 size/codes/IDs 视图上扫描。
条目必须在进入 candidate heap 之前满足 row ID `< visible_hi` 与 ready/epoch 条件；
事后删除 future IDs 会影响合法候选的名额，不算等价 snapshot。更新 centroid/codebook 时
构建新 owner，不能与查询并发修改旧参数。Faiss 控制保持原地串行锁，不解除其同步要求。

追加记录经历“GPU 可用 → 复制中 → backing 可用 → 索引条目初始化完成 → 发布”。先写 payload、ID 和编码，再发布有效标志/已提交长度；未完成条目不可被搜索。多条并行写入时，共同水位只推进完整前缀，不能以最大已完成 ID 代替。

查询返回以下两部分的候选：已发布 base/delta 的 ANN 候选，以及尚未入索引尾部的精确结果。sink/window/current token 可直接进入 mandatory 路径；与索引命中、tail 交集按逐 head ID 去重。任何不在 ANN 可见前缀内的合法 KV 都必须由精确尾部覆盖，不能只覆盖最近窗口而遗漏排队数据。

pending tail、追加日志和复制队列均有上限。维护或复制跟不上时，暂停准入、限制进一步生成或同步推进，并记录等待；不能丢 KV、提前复用槽位或无限扩大 tail。CXL 尚未 ready 的条目必须保留可读取的 GPU/DRAM 来源，不能释放唯一副本。

delta flush 按批量/时间间隔触发，流程为：pin 要插入的 K → 按 A 参数 assignment/encoding →
初始化并发布条目 → 整体更新 indexed prefix 与 delta 描述。条目完全可见后，新的查询才可从
delta 移除相应 ID；旧查询 pin 的 delta 内存等读者结束后回收。切换期间两处同时包含某 ID
时按 head 去重，不能出现覆盖空洞。本文水位使用半开区间：`[0,indexed_hi)` 已入索引，
`[indexed_hi,visible_hi)` 为需完整覆盖的 delta/tail。精确 delta 的数据和重建追赶日志是
不同对象，后者保存 ID/K 引用与顺序，不能代替可读取的 delta K。

## 4. 替换、追赶与发布

MVP 采用单 writer、一个 builder；仅在 turn 的 prefill 开始时进行 rebuild admission。
下面的 build/catch-up/publish 在该 prefill 阶段完成；同 session 的 prefill 与 decode 顺序执行，
不假设它们同时发生。CPU builder 可以和 GPU 新 suffix 的 prefill 重叠，但实际竞争与等待计时。
更新提交与发布由同一短临界区协调，索引长扫描不持有发布锁。

1. 下一轮 prefill 开始时取得已有历史的完整 build 水位 `b`，pin 对应 canonical K 目录，包括尚在 delta 的已提交 K。活动索引 A 保留有效；本次 prefill 新产生的 K 同时进入有界日志。
2. 在空闲槽位构建 B，覆盖前缀 `[0,b)`。同 centroid 的 compaction 与重新训练是不同操作；重新训练可能需要 K 扫描、重新 assignment 和编码，全部计成本。
3. 在锁外反复追赶 `[b,c)` 的本次 prefill 新增记录，直到剩余 backlog 小于预设切换预算。若 B 参数改变，不能直接拼接 A 的 residual codes；从原始 K 按 B 参数重新生成条目。
4. 进入有界提交门，停止向 A 追加并冻结其增量描述，选择完整 cutover 水位 `c`，补完最后一批，建立“B 覆盖至 c + B 的新 tail/delta”这一完整 manifest。期间新产生的 KV 留在受保护的 pending tail，相关等待和精确扫描必须计时。
5. 以 release/acquire 发布 manifest 指针与 generation。发布是一个整体操作，不能分别切换 centroid、postings 和水位。后续查询 pin B，已有查询仍使用 A 和自己的可见上界。
6. A 进入退役状态；实际借用 A 索引存储的读者/任务完成后，才归还其内存块。独立的 K/V、hot cache、staging 与 partial owner 等各自 CPU/CUDA/DMA/安装事件，不共用索引槽的释放条件。构建失败时放弃 B，保留完整 A + delta，失败的时间和内存仍进入记录。

publication 并不意味着零暂停；应分别报告构建、catch-up、临界区、reader drain、可重新利用时间和新轮 TTFT。初版在 prefill→decode 边界等待已获准 B 完成；构建失败则取消 B 并保留完整 A + delta。退役回收可以延后到 decode，但不启动或继续全量建造。若最终追赶无法在预算内收敛，采用显式回压；更细的分区替换、并行多编码 delta 属于后续方案。

## 5. 索引槽位环与真正可复用的资源

槽位状态为：

```text
空闲 → 构建中 → 待发布 → 活动 → 退役 → 可回收 → 空闲
```

两个槽位支持“一个活动、一个构建”；活动切换后，第二槽位可能仍被旧读者占用，因此不能立即启动下一次 rebuild。第三槽位能缓解 build/serve/retire 重叠，但增加峰值容量。槽位复用同时更新 generation，避免旧 handle 指向同一地址上的新索引，即 ABA 问题。

prefill 按 layer 产生 K/V，为逐层 assignment/编码和 scratch 复用提供机会。层间依赖、offload 完成事件和共享带宽决定是否真能重叠；构建仍会增加首轮或新轮 TTFT。每层/group 的活动索引必须留存，不能拿一对全局 buffer 循环覆盖所有层。第一版只允许一个 layer/group 的额外替换索引同时构建，用全局 credits 控制 scratch 与临时容量。

索引槽位环与 H2D packet/staging 环是两个池：前者等所有索引读者结束，后者等传输及 attention 消费结束。不可使用相同完成条件。allocator 保留内存用于复用时，“逻辑释放”也不等于 RSS 或物理容量下降。

逻辑清空可用 generation、有效长度和块描述复位，避免逐元素归零；旧条目必须被新有效范围隔离。若 `free()`、析构、遍历元数据或设备端 reset 仍是 O(N)，不能只测 pointer swap 宣称整体 O(1)。

## 6. 容量与布局取舍

估算须乘上所有 layer/group。设 L 层、G 个 KV heads、N tokens、维度 d、每元素 b bytes：

- canonical KV：`2 × L × G × N × d × b`。
- 紧凑索引：约 `L × G × N × (ID bytes + code bytes)`，另加 centroids、metadata、slab slack、allocator 和 build scratch。
- 峰值：所有活动索引 + 有界替换索引 + 仍未回收的旧版本 + delta/log/tail + canonical KV + resident 副本 + staging/workspace。

例如 L=32、G=8、d=128、BF16、N=1,000,000，canonical KV 约 122.1 GiB；若每向量存 4-byte ID 与 32-byte code，所有活动索引约 8.58 GiB，一个 layer 的额外索引约 274.7 MiB，均不含额外开销。64-bit ID 须按 8 bytes 计费；另存 BF16 K 约 61.0 GiB，标准 CPU Faiss 的额外 **FP32 K** 则约 122.1 GiB，另加 IDs/lists/workspace。HAKES filter-only 不启用全量 refine copy，但 codes/metadata/alignment 和临时分配仍须逐项记录。以上仅为架构示例和容量推算，不是测量结果。

32-byte PQ code 不是最优参数承诺；MVP 的 IVF-Flat 可用 ID-only postings 扫 canonical K，随后加入 SQ/PQ。连续 postings/codes 不意味着 canonical K 连续：保持 token-major 稳定 backing 会引入随机 gather。先测按物理地址排序和合并 extent；cluster-major canonical 布局另设消融，不能无成本继承 InSituANN 原向量重排的局部性收益。

## 7. 维护决策与验证问题

轮内只比较固定参数查询、nprobe/候选量调整与定期 delta flush；换 centroid/codebook、全量重建及需要重组织历史的 split/merge 仅在下一轮 prefill 开始时决定并启动。边界策略比较永不重建、每轮 prefill 开始重建、按过去统计选择部分 turn 重建。Q 分布漂移、K 新增分布和 posting skew 是不同信号；topic switch 不自动意味着必须 rebuild。

成本决策使用已发生的查询、写入、队列和 sampled audit，以及预测的剩余 session 查询量。用未来 Q 或事后最佳策略只构造离线 oracle。在线 exact recall audit 需要读完整 K，其成本必须收费；不能把离线完整 attention mass 当作免费在线触发器。

先比较二/三槽、每 turn 独立索引与全局 base+delta、new allocation 与槽位复用、logical reset 与逐项清空。报告 query/insert P50/P99、质量随 turn 变化、backlog、K 扫描、维护带宽、peak live/reserved memory、reader drain 和整个 session 摊销成本。质量、生命周期正确性、执行效率和 CXL 硬件结果分别验证。

现有代码只提供 request 级 DynamicCache observer 和 tensor borrow lease，尚不提供这些保证。multi-turn 的最后一个已输出但尚未前向的 token、EOS、chat-template 后缀必须先正确并入缓存，具体接口约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。
