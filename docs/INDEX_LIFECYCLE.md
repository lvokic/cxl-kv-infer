# 多轮 IVF 索引生命周期：低成本追加、替换与内存复用

更新：2026-09-30。本文是目标协议与研究假设；持续 session、IVF、版本槽位和 native allocator 尚未实现。研究叙事见 [PROPOSAL.md](PROPOSAL.md)，组件契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 核心想法与适用范围

在一个持续多轮 session 中，prefill 分批产生新 K/V，decode 逐 token 产生新 K/V。累计 context 的本阶段目标上限约为 1,000,000 tokens；它是研究 workload 的规模，不是所有模型的最大能力。历史 KV 不随 turn 结束清空，向量索引持续追加、复用和维护。

希望把常见路径做成便宜的 append，把昂贵维护移到另一个槽位：当前索引继续服务，新索引在后台建立，准备好后切换活动指针，旧查询完成后回收旧槽位，并在下一次维护时复用。环形槽位管理分配与版本生命周期；发布开销可以与索引大小无关，构建、追赶写入、等待读者和清理的总成本仍需计量。

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

起点为不可变 IVF base + 可搜索增量区 + 精确 pending tail。增量区内条目使用当前 generation 的 routing/encoding 参数；首批数据不足以训练时先走 exact tail，训练只消费当前已可用 K。

追加记录经历“GPU 可用 → 复制中 → backing 可用 → 索引条目初始化完成 → 发布”。先写 payload、ID 和编码，再发布有效标志/已提交长度；未完成条目不可被搜索。多条并行写入时，共同水位只推进完整前缀，不能以最大已完成 ID 代替。

查询返回以下两部分的候选：已发布 base/delta 的 ANN 候选，以及尚未入索引尾部的精确结果。sink/window/current token 可直接进入 mandatory 路径；与索引命中、tail 交集按逐 head ID 去重。任何不在 ANN 可见前缀内的合法 KV 都必须由精确尾部覆盖，不能只覆盖最近窗口而遗漏排队数据。

pending tail、追加日志和复制队列均有上限。维护或复制跟不上时，暂停准入、限制进一步生成或同步推进，并记录等待；不能丢 KV、提前复用槽位或无限扩大 tail。CXL 尚未 ready 的条目必须保留可读取的 GPU/DRAM 来源，不能释放唯一副本。

## 4. 替换、追赶与发布

MVP 采用单 writer、一个后台 builder；更新提交与发布由同一短临界区协调，索引长扫描不持有发布锁。

1. 取得完整的 build 水位 `b`，pin 对应 canonical K 目录。活动索引 A 继续提供读服务；新写入同时进入有界日志。
2. 在空闲槽位构建 B，覆盖前缀 `[0,b]`。同 centroid 的 compaction 与重新训练是不同操作；重新训练可能需要 K 扫描、重新 assignment 和编码，全部计成本。
3. 在锁外反复追赶 `(b,c]` 的新增记录，直到剩余 backlog 小于预设切换预算。若 B 参数改变，不能直接拼接 A 的 residual codes；从原始 K 按 B 参数重新生成条目。
4. 进入有界提交门，冻结 A 的增量描述，选择完整 cutover 水位 `c`，补完最后一批，建立“B 覆盖至 c + B 的新 tail/delta”这一完整 manifest。期间新产生的 KV 留在受保护的 pending tail，相关等待和精确扫描必须计时。
5. 以 release/acquire 发布 manifest 指针与 generation。发布是一个整体操作，不能分别切换 centroid、postings 和水位。后续查询 pin B，已有查询仍使用 A 和自己的可见上界。
6. A 进入退役状态；所有读者、CPU task、CUDA kernel、DMA、cache install 和 build lease 完成后，才归还其内存块。构建失败时放弃 B，A 继续服务，失败的时间和内存仍进入记录。

publication 并不意味着零暂停；应分别报告构建、catch-up、临界区、reader drain、可重新利用时间。若最终追赶无法在预算内收敛，采用显式回压或更细的分区替换。并行多编码 delta 属于后续方案，MVP 不靠它掩盖追赶成本。

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

例如 L=32、G=8、d=128、BF16、N=1,000,000，canonical KV 约 122.1 GiB；若每向量存 4-byte ID 与 32-byte code，所有活动索引约 8.58 GiB，一个 layer 的额外索引约 274.7 MiB，均不含额外开销。若采用 Faiss 64-bit ID，要改用 8 bytes；若 IVF-Flat 另存完整 K，复制开销可接近完整 K 的约 61.0 GiB。以上仅为架构示例和容量推算，不是测量结果。

32-byte PQ code 不是最优参数承诺；MVP 的 IVF-Flat 可用 ID-only postings 扫 canonical K，随后加入 SQ/PQ。连续 postings/codes 不意味着 canonical K 连续：保持 token-major 稳定 backing 会引入随机 gather。先测按物理地址排序和合并 extent；cluster-major canonical 布局另设消融，不能无成本继承 InSituANN 原向量重排的局部性收益。

## 7. 维护决策与验证问题

由轻到重比较：保持现状、调整 nprobe/候选量、同 centroid 合并 delta、分区 split/merge、重训 centroid/codebook、全索引替换。Q 分布漂移、K 新增分布和 posting skew 是不同信号；topic switch 不自动意味着必须 rebuild。

成本决策使用已发生的查询、写入、队列和 sampled audit，以及预测的剩余 session 查询量。用未来 Q 或事后最佳策略只构造离线 oracle。在线 exact recall audit 需要读完整 K，其成本必须收费；不能把离线完整 attention mass 当作免费在线触发器。

先比较二/三槽、每 turn 独立索引与全局 base+delta、new allocation 与槽位复用、logical reset 与逐项清空。报告 query/insert P50/P99、质量随 turn 变化、backlog、K 扫描、维护带宽、peak live/reserved memory、reader drain 和整个 session 摊销成本。质量、生命周期正确性、执行效率和 CXL 硬件结果分别验证。

现有代码只提供 request 级 DynamicCache observer 和 tensor borrow lease，尚不提供这些保证。multi-turn 的最后一个已输出但尚未前向的 token、EOS、chat-template 后缀必须先正确并入缓存，具体接口约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。
