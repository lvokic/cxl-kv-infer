# 面向多轮超长上下文的持续 IVF 检索与分层 partial attention 系统

## 1. 背景与动机

### 1.1 从固定上下文检索走向持续多轮 KV

长上下文推理需要保留随输入与生成增长的 K/V。训练后 sparse decode 通过向量检索选择部分历史 token，能减少一次 attention 的读取与计算，但完整 KV、索引、缓存与临时缓冲仍占容量。多轮对话使问题更加具体：每个 turn 引入一批 prefill K/V，assistant decode 逐 token 追加，后续查询继续访问早期历史；topic、访问热点和 list 分布会随 session 改变。

因此，研究对象是整个 session 的索引与 attention 生命周期。只在固定 prompt 上建一次索引，随后仅报告 decode 延迟，会隐藏新轮追加、索引维护、峰值分配和老版本回收的代价。应同时解释：插入是否足够便宜、索引何时仍值得复用、何时更换，以及维护是否挤占真正需要的查询带宽。

### 1.2 为什么选择 IVF 与替换式维护

IVF 将 coarse routing、list 扫描和候选精排分开，方便控制新条目的 assignment/encoding 与 list append，也方便把 codes、ID 和原 K 分层放置。它并不保证优于图索引：attention 的 Q/K 分布差异、偏斜、较高 nprobe、分散 K gather 和小批开销都可能使其变慢。当前主线先选择 IVF，以隔离持续更新和内存生命周期；图索引暂不作为核心开发。

一个可探索的机制是：不在在线索引上“清空再重建”，而是在另一个已预留槽位建立新版本，追赶写入后切换 manifest 指针，等旧查询结束再回收。索引槽位以环形方式复用，减少反复分配和大范围 reset。prefill 逐 layer 产生 K/V，又为逐层建索引和 scratch 复用提供调度机会。

这些机制有先例：HAKES 涉及 CPU 并发插入与 filter-refine，SIVF 涉及 GPU streaming append 的发布安全，InSituANN 提供 snapshot/delta 与替换式维护参照，Quake 研究自适应维护。[R1–R4] 本研究要验证它们在多轮、小批 Q、逐 token 更新和 attention 质量约束下的适配，而不是把指针切换本身包装成贡献。

### 1.3 为什么执行与内存管理必须一起研究

索引找到 token 后，仍需读取原 K 精排、获取 V、执行 softmax/PV 并归并 partial。检索阶段已经获得的 K 或 exact logits，若在执行阶段丢弃，会产生重复访问；GQA heads 的选集重叠又可能提供共享物理读机会。与此同时，索引双缓冲、candidate K、HBM cache、pinned packet 和 GPU workspace 必须争用同一资源预算。

我们以三个相互约束的目标开展研究：

| 目标 | 操作化定义 |
|---|---|
| 有效性 | 在预先登记的质量要求下，检索召回、attention 误差和多轮任务质量足够好；正确保留 KV、position 和逐 head mask |
| 高效率 | 降低整个 session 的构建/插入/维护/查询与执行成本；以端到端 TTFT、ITL/TPOT、P99 和实际阶段时间衡量 |
| 合理内存管理 | 有界 active/building/retired index、tail、staging 和 HBM 副本；稳定 ownership、安全复用和明确 DRAM/CXL 驻留证据 |

GPU kernel 可以进入实现，但前提是实测表明 gather、masked GQA、softmax/PV 或 merge 为热点。原 K 带宽或索引维护占主导时，加速一个 kernel 不一定改善整个 session。

## 2. 研究边界

主场景是单 session 多轮追加，并逐步加入少量独立 sessions 的竞争；起步为单模型、单层、单 KV group。context 指 session 累计已提交的 tokens，本阶段考察 64K、256K 和约 1M 的索引 workload；真实模型仅在其已验证 position/RoPE 范围内运行。

| 包含 | 本阶段边界 |
|---|---|
| post-training sparse decode、真实 post-RoPE Q/K、标准 scaled inner product | 不训练新 sparse 模型，不把 cosine 或 PQ 分数当 attention logits |
| IVF-Flat 控制、紧凑 SQ/PQ 候选、exact K refine、partial attention | 先固定 selector 语义，再分别测维护、执行优化 |
| session 级 append、compact/retrain/replacement、索引槽位池与有界回压 | 首轮单 writer，无通用事务、crash durability 或并发图 mutation |
| 原始 K/V 分离后备、GPU 热区、DRAM 索引、真实 CXL placement | memory-only CXL 不运行 compute；GPU direct/mapped access 为可选独立路径 |
| dense causal 增量 prefill 与必要的有界读取/缓存协作 | 不把新 prefill 稀疏算法作为核心；当前 runner 尚不支持长历史 streaming prefill |
| 固定选集的 CPU/GPU 执行对照；测量后考虑 CUDA/SIMD 热点 | 不重建 serving engine；多 GPU/PP、跨租户 prefix sharing、KV 淘汰留待后续 |

1M 索引持续性不等于 1M 模型能力。真实 1M 端到端依赖适合的 checkpoint、模型 adapter、RoPE 契约和 chunked/offloaded prefill；现有 Llama-family adapter 不能直接接任意模型。模型/benchmark 均本地运行，固定 revision 或文件散列，不依赖远程推理 API。

## 3. 研究问题与预期学术贡献

### 3.1 可证伪问题

| 问题 | 核心假设 | 可能推翻它的结果 |
|---|---|---|
| Q1：何时复用，何时维护？ | 生命周期决策能比固定每 turn rebuild 或永不 rebuild 更便宜地保持质量 | 固定 centroid + 调 nprobe 已同样有效，rebuild 无法摊销 |
| Q2：索引替换能否稳定持续更新？ | layer-aware 槽位复用、追赶和 reader retirement 可限制暂停/峰值 | 追赶无法收敛、退役过慢或内存预算不可满足 |
| Q3：检索与执行的交接是否浪费？ | KLease/exact-logit handoff 可减少重复 K 读取与 H2D | 额外租约、状态传输和同步吃掉收益 |
| Q4：何时值得优化 GPU kernel/分流？ | 同一 mask 下物理 span、GQA overlap 和 residency 能指示更快路径 | whole-group/成熟 FlashInfer 已接近最优，mask/merge 开销更大 |

研究优先级是 Q1/Q2；Q3 构成共同数据路径，Q4 根据测量决定投入。不是要求第一篇同时完成新的索引算法、planner 和通用 attention kernel。

### 3.2 候选贡献及证据要求

1. **多轮 attention 索引生命周期的经验模型与决策机制。** 区分 Q drift、K drift、list skew 和 query-count horizon，选择继续复用、调 probe、compact 或 retrain。要在相同质量和资源下优于强固定策略，并报告审计/预测成本。
2. **在质量和内存预算内的替换式维护。** 研究逐层 build、受限版本槽位、可见水位与追赶、异步 retirement 的协作，使持续插入、query P99 和峰值容量可解释。RCU、双缓冲、ring 和内存池本身均不是新概念。
3. **检索到 partial attention 的高效物理交接。** 保持逐 head 语义，复用 exact K/logits，晚获取 V；若 GPU kernel 或有限混合分流提供端到端净收益，再作为效率贡献。

预期学术主张应来自“何时有效、为什么有效、在什么边界失效”的结果，不来自“IVF + CXL + attention”的组合名称。若只有前两项成立，保留成熟 executor 是完整而合理的研究结果。

## 4. 技术方案

### 4.1 系统 architecture

![目标系统架构：持续 IVF、分层内存与 partial attention](figures/architecture.png)

图 1 是计划实现的系统架构；已有实现和待实现部分的对应表见 [ARCHITECTURE.md](CXL_KV_Architecture_v2.html#arch-1)。图中 online path 与 maintenance path 共享预算，但有不同的 owner/完成事件。

全英文图中的五个面板对应以下职责；正文以中文解释模块如何协作：

| 面板 | 阅读重点与责任边界 |
|---|---|
| (a) System Framework | GPU HBM 保存权重、有限热 KV 和执行缓冲；CPU/local DRAM 保存索引、目录与控制状态；memory-only CXL 保存冷 K/V。生命周期与预算管理器协调在线读、维护和回收。冷数据经 CPU 读取与 staging 后送 GPU，或由 CPU 执行 partial。 |
| (b) Layer-wise Build | 每层产生 K 后做 assignment/encoding，建立该层/group 的持久索引；V 进入独立 backing。构建 scratch 按层复用，活动索引不能被下一层覆盖；所有构建时间与流量计入 TTFT。 |
| (c) Query & Partial Attention | Q 查询 IVF，候选经 canonical K 精排成为逐 head CSR 选集，再获取选中 V。KLease/exact logits 可交接；GPU/CPU partial 汇入同一 merge，输出 normalized output 与自然对数 LSE。精确 tail 覆盖全部尚未 indexed 的合法 KV。 |
| (d) Incremental Append | 初始化 K/V 与条目后发布；只有 K 用于索引 assignment/encoding，V 保持独立所有权。append log 与有界 pending tail 保护新数据和复制中条目；共同水位不跨过未完成 hole。 |
| (e) Replace & Reclaim | 在空闲槽构建新索引并追赶日志，以完整 manifest 切换活动版本。旧版本等 reader、CPU task、CUDA/DMA 与安装事件结束后复用；切换指针不免除 build/catch-up/drain 的成本。 |

| 组件 | 特定问题领域 | 主要功能与协作 |
|---|---|---|
| 持续 Session / ModelAdapter | 多轮语义与 KV 重用 | 保留 cache，验证 chat prefix，追加 suffix，提供真实 Q、new K/V、position 和事件 |
| KV owner / backing 目录 | 大容量存储与一致地址 | K/V 分离，稳定 token IDs，记录 tier/地址/水位，提供 lease 与安全回收 |
| IVF selector | 动态向量检索 | coarse routing、codes/list 扫描、逐 head candidates；原始 scaled IP 精排前不声明 exact |
| 维护控制器 + 索引槽位池 | reuse/rebuild、峰值与暂停 | 观察过去统计，限制并发 rebuild，追赶 tail，原子发布 manifest，延迟复用旧槽 |
| K 精排与 Selection | 检索误差与重复读 | candidate union 批量读取 K、per-head refine；交接选中 K lease / 真 logits |
| Work/credits 规划 | 稀疏访问、带宽竞争 | 根据实际 residency、span、membership 形成有限 routes，约束读取/传输/维护 |
| CPU/GPU partial executors | attention 核心执行 | GPU resident/packet、CPU historical partial；输出自然对数 LSE 与 normalized output |
| 测量与 artifact | 证据和可复现性 | 保存源码版本、模型、stream、placement、timing、bytes、quality 与失败记录 |

拟议实现沿现有 `src/module/` 扩展模型/session 接入；后续按实际实现增加 `src/index/`、`src/storage/`、`src/retrieval/` 和 `src/execution/`。Python/NumPy 做 reference/control，Faiss 为索引对照；native/CUDA 仅用于测得的热点。不会仅为文档创建空目录。

### 4.2 Session、prefill 与 append

同一 session 横跨 user/assistant turns；不重放历史文本。完整 chat-template token prefix 必须与已提交 cache 匹配，验证后追加新 suffix；position 单调延续、causal mask 以全历史长度为基准。现有 greedy loop 最后一个 emitted token 尚未前向，目标 runner 显式管理 pending token、EOS 与模板后缀，在下一 turn 前提交且避免重复。

prefill 在各层产生 K/V 后触发 index append/build，可复用该层 construction scratch 并与 offload 协作。索引 build 并非免费，可能争用模型计算和链路；记录其对首轮、新轮 TTFT 的影响。真实长历史 prefill 必须有 dense causal 的有界 chunking/streaming 或成熟后端支撑，不能只凭 index 到 1M 宣称模型可运行。

### 4.3 IVF 表示、增量与质量

首先实现 ID-only IVF-Flat list 控制：索引只存 ID、centroid/list 目录，精确扫描指向 canonical K；Faiss 全量 Flat/IVF-Flat 另作对照，记录其向量复制和实际 ID 宽度。随后加入 SQ/PQ codes，候选只负责过滤，final ranking 一律来自真实 post-RoPE K。

每个 layer/KV group 建一套索引，Q heads 共用 K/codes，保持独立 candidates 和 top-k。training/assignment/query metric 必须一致并记录；centroid 的 routing normalization 不得改变 canonical K 或原 attention IP。跨 session 复用 centroid/codebook 是独立消融，不能默认无损。

bootstrap 使用当前可用 K；未训练、尚未 indexed 或尚未 backed 的合法数据使用 exact tail。pending tail 必须有界且完整覆盖未发布后缀；sink/window/current-token 与历史命中交集去重。扩大 nprobe 只能改善被访问候选，不是未访问 token 质量的证书。

### 4.4 索引槽位、追赶与 pointer swap

每个空间拥有活动 manifest，绑定 centroid/codebook、base、delta/tail 描述、KV 布局版本和可见 token 水位。reader pin 与 writer publication 使用安全协议；MVP 用短锁、单 writer 和显式 reader leases。

replacement slot 先构建完整前缀，再追赶 build 期间的 append log。在有界提交门确定 cutover，完成 catch-up 并整体发布新 manifest。若 codebook/centroid 改变，新增向量必须按新参数编码，不能直接拼接旧 codes。旧读者继续使用旧 version；CUDA、DMA、CPU task 和 cache install 完成后才可回收。失败构建不影响活动版本。

二/三槽复用只降低分配/reset 开销，不使构建、catch-up 或 reader drain 成为零。每层/group 的活动索引都要保留；优先全局限制“一次额外构建一个 layer/group”，scratch 跨层复用。逻辑清空使用 generation/valid length 隔离旧条目，重复地址使用 generation 防止 ABA。详见 [INDEX_LIFECYCLE.md](CXL_KV_Architecture_v2.html#lifecycle)。

### 4.5 内存管理与布局

以稳定 row ID 和 token-major canonical K/V 起步，K/V 独立所有权与放置。codes/list IDs 在 DRAM 连续，并不保证对应原 K 在 CXL 连续；原 K gather 的分散度必须实测。先尝试 union、physical-address 排序、extent 合并；list-major payload 或 sealing-time 布局重排是独立消融，计扫描、复制、映射和旧版本成本。

例：32 层、8 KV heads、d=128、BF16、1M tokens 的 canonical KV 约 122.1 GiB；4-byte ID+32-byte code 的所有活动索引约 8.58 GiB，一个 layer 的替换 payload 约 274.7 MiB。ID-only Flat、64-bit IDs、full-vector Flat 复制会改变这个数；算式与额外项目见 lifecycle。有限规模与紧凑索引使 layer-wise replacement 有可行性，但这些数字不是“索引总是很小”的证据。

allocator 分别管理 index slot pool、KV extents、candidate K pool、pinned/H2D packet ring 和 partial workspace。记录 logical live、allocated/reserved、RSS、fragmentation/slack、peak 和真正可复用的时间。HBM budget 先扣权重和模型 workspace；DRAM budget 同时容纳 active+retired index、builder、staging；CXL budget 包含 canonical payload 与后备余量。释放 owner 不等于释放物理容量。

平台初始记录为 A100-PCIE-40GB，GPU-local node 1，CXL memory-only node 3。硬件运行前重新核对 topology/驻留。基线路径是 CPU 读取 DRAM/CXL 后 pinned staging→PCIe→GPU，或 CPU 原地 partial；不假设 GPU 原生直接访问 CXL。只绑定具体 allocation 与 worker，不把整个进程绑到 CXL node。

### 4.6 Partial attention、K/logits 复用与 GPU kernel

选择结果 S_h 和物理 union 分开；所有 routes 对每个 head 的贡献两两不交、并集恰好为 S_h。候选 K 精排后保留 selected KLease；只在确定 S_h 后取 V。exact logits identity 包含 session、step、head、layer/group、position/score transform、scale 和 precision，近似 scores 不能进入 softmax。

内部统一为 `(normalized output, natural-log LSE)`；backend 的不同 LSE 底由 adapter 转换。CPU/GPU hot/cold 分区的归并使用稳定 logaddexp，并处理空分区。数学契约与实现证据见 architecture；不是把当前 online output-only executor 当成已实现的跨设备 partial 系统。

先实现固定 GPU K+V packet 和优化 CPU whole-group 控制。按热点再比较：

- exact logits + selected V 的 GPU softmax/PV，减少 K 再上传与 QK 再算。
- masked GQA shared-KV tile，保留 per-head membership，减少当前独立逻辑行可能造成的重复加载。
- 有限批量 gather/descriptor/merge kernel，减少 launch 和 intermediate materialization。

kernel 输入使用有界 SoA/CSR/bitmap，post-RoPE K 不按 packet offset 再做 RoPE；padding 不进 softmax。SIMT、Tensor Core、tile 和 signature packing 通过 shape profile 决定，不能预设赢家。

### 4.7 三条路径的协调与维护策略

查询和必要 KV 读取优先，预取使用剩余 credits；维护按片运行并给有限进度保证。维护与查询共享 CXL/DRAM/PCIe 带宽，不能分别按独占峰值建成本模型。没有空闲 slot 或 tail 达预算时显式回压，记录是否影响生成。

维护动作按成本由轻到重：调 probe/候选、同参数 compaction、partition 局部维护、centroid/codebook retrain、全 replacement。观测 Q 分布、K assignment residual、list skew、候选量与 tail/backlog；topic switch 不自动触发 full rebuild。在线 recall audit 如需 exact scan，单独收费。

决策目标是满足质量、P99 和三层容量约束时，降低预测剩余生命周期成本。只使用已发生 queries 和 past statistics 预测剩余查询量；未来 Q 与事后最优维护点只作为 diagnostic oracle。不保证索引维护始终能摊销，保留直接 scan 和固定参数路线。

### 4.8 已有代码与实验提供的起点

当前 repo HEAD 为 `60e69caa3d5aff645dae29eec938ab36df86ba64`，文档修订为其上的工作区变化。现有 ModelAdapter、KVCacheLifecycle、TokenSelector、AttentionExecutor 和 trace/reference 已提供可扩展边界，但 DynamicCache 借用不是 native stable store，completion 参数也没有真实等待实现。

既有 CPU numerical run 为 482 项，GPU numerical run 为 644 项且包含 BF16/FP16 与 reference/contract 检查，不应把 644 当作每种 dtype 独立总数。真实固定 revision SmolLM2 的 FP16 最新整理检查 20/20、FP32 18/18，BF16 16/20，四项严格 logits 对照失败。所有结果用于数值/集成，不提供任务质量、动态索引或硬件性能证据。[RESULTS.md](../../../cxl-kv-replay/docs/RESULTS.md)

原始 manifest/source.zip 保留运行时 dirty 状态、源码散列和依赖，不能仅凭当前 HEAD 代表旧实验。已记录环境包括 Python 3.12.14、NumPy 2.5.3、PyTorch 2.14.0+cu130、Transformers 4.57.6、FlashInfer 0.7.0；这是既有运行记录，不是本次重新安装或 benchmark。

## 5. 实验设计

### 5.1 工作负载与模型

第一类是可控向量 stream，用于 1M scale 的插入、并发 visibility、skew 和 version churn，明确标记 synthetic。第二类是真实 session Q/K/V trace，用于相同输入下隔离 index、precision、layout 和 execution。第三类是自由生成多轮任务，测误差累积和最终质量，不能仅用 teacher forcing 代替。

multi-turn trace 记录每轮新增文本 token、assistant 输出 token、主题切换/回访、当前 position、insert/query 顺序、已提交水位、arrival、turn think time、寿命和取消。固定总 token 数时改变 turn 数/批量大小；长初始 prefill+少轮、短 turn+多轮、主题持续/切换/回访、长生成分别测试。单 session 的 query/insert 有因果依赖，不能以任意 QPS 比例模拟实际 decode。

模型从 SmolLM2 数值控制扩到本地可用 GQA Llama-family 模型；revision、RoPE 和 context limit 先核查，不预先声称 256K/1M 都可跑。真正 1M checkpoint 若需新 architecture/RoPE adapter，作为明确依赖和独立实现工作登记。

[R11] RULER 提供可控长上下文任务，[R12] LongBench 提供真实任务质量参照；选取适合模型长度的 QA、摘要和多针/聚合任务。将信息分批跨 turns 注入、后续追问旧事实/最新修正/跨轮聚合，属于**本项目派生 multi-turn workload**，不冒充原 benchmark 官方分数。派生数据、评分、seed 和 tokenizer 固定并公开。

### 5.2 分阶段矩阵

| 阶段 | 固定条件与比较 | 必须保存的证据 |
|---|---|---|
| E0：session/生命周期正确性 | 单层/group；base+delta 对 FlatIP；跨 turn template、pending token、current token、holes、reader retirement、失败 build、ABA | 完整 event trace、visible prefix/ID 覆盖、leases、数值与错误记录 |
| E1：reuse/rebuild | 相同 Q/K stream、质量、cores 和容量；never rebuild、每 turn、固定 delta/年龄、同 centroid merge、periodic retrain、adaptive；全局 vs per-turn segment | 每 turn recall、nprobe、候选量、skew、插入/查询 P50/P99、build/catch-up/drain 和 session 摊销 |
| E2：内存管理 | 二/三槽 vs new allocation；逐层 vs 并发 rebuild；logical reset vs 逐项 clear；稳定 token-major vs 受控布局 | peak live/reserved/RSS、allocator churn、scratch、backlog、ready/可复用事件、流量 |
| E3：有效性 | full→exact top-k→IVF Flat→compressed IVF，同模型/精度与 mandatory policy | top-k recall、保留 attention mass、output/LSE/logit error、逐 turn 任务分数与 free generation |
| E4：执行效率 | 冻结同一 S_h；whole-group CPU/GPU、hot+history、K/logit reuse、成熟 kernel 与候选 CUDA kernel | Q-ready→output-ready、gather/H2D/plan/run/merge/join、GQA 复用、CPU 占用和 GPU profile |
| E5：真实内存层次/端到端 | local DRAM、remote DRAM、CXL 同 selector/资源；equal-capacity/equal-batch 后再做 DRAM 预算约束 | 页驻留/topology、实测 counters 或清楚命名的逻辑 bytes、TTFT/ITL/TPOT、turn P99、总 session 成本 |

先 E0，再 E1/E2，E3 决定可接受 retrieval budget，E4 优化热点，E5 才形成 CXL/模型系统结论。所有验收条件为拟议方案，本次文档更新不新增或执行测试。

### 5.3 强基线与公平性

索引比较包含 locked Faiss IVF（调用方负责 add/search 同步）、exact scan、snapshot/delta 固定维护、HAKES 风格并发 filter-refine；InSituANN 风格 replacement 和 Quake 风格动态 partition 为维护参照。[R1,R3,R4,R5] SIVF 仅在 GPU-resident 条件匹配时作为执行 baseline，否则只作协议参考。[R2]

固定选集执行基线包含 FlashInfer selected K/V、优化 CPU partial、直接 packet attention、固定 hot/history 分工、静态 whole-group profile。RetroInfer、RetrievalAttention、Fluxion、HiSparse 的机制已覆盖多项相邻能力，具体差异在第 8 节；未运行上游 artifact 时写“机制复现”，不可写成击败原系统。[R6–R10]

保持模型、revision、dtype、CPU cores、HBM/DRAM/CXL 总预算、候选/质量要求一致；索引复制、audit、build、tail、maintenance 和失败恢复均计费。offline oracle 使用相同资源、只提供上界。等 batch 与受限容量下最大可行 batch 分开报告，不用多余内存换取虚假速度。

### 5.4 指标、归因与复现

三个误差轴分别报告：full→exact top-k 截断、exact top-k→IVF 路由、IVF Flat→compressed code 额外损失。exact rerank 不能恢复未进入候选的 K；recall 也不充分代表任务质量。逐 head 的 retained softmax mass 在离线 full oracle 中计算，MVP 不拿它当免费在线信号。

效率指标含首轮/新轮 TTFT、每 token ITL 的 P50/P99、请求平均 TPOT 的 P99、整个 session wall time、有效 token/s、维护暂停和回压。区分 query-only kernel、attention 完整路径和整模型；观察 MLP/权重/prefill 是否限制可优化比例。

内存和流量含 active/building/retired index、KV、tail/log、codes/IDs、candidate K、staging、workspaces、copy bytes、read amplification、peak/reserved 和碎片。硬件 counters 不可用时写 logical_payload_bytes / estimated_transaction_bytes，不写 measured CXL bytes。CXL 容量价值必须在明确 DRAM budget 下测试。

timing 保存 CPU monotonic、CUDA events 和统一范围；不直接拼接不同时钟，不用无条件 max 假设完美重叠。setup、JIT、冷/热 cache、profile 和 capture 单列；多次独立运行报告分布与不确定性，质量阈值在测试集前登记。

每个新 run 保存源码 commit+dirty hash/source snapshot、模型文件/revision、dependency versions、配置/seed、workload stream、position/score 契约、topology/页驻留、结果与失败。保留所有既有输入和输出，不覆盖原目录。

## 6. 预期结果与停止条件

预期产出是可验证的多轮 IVF/partial attention prototype、公开 workload/event trace、维护策略的生命周期成本曲线，以及 memory budget 与执行路径的适用区域，而不是预先指定加速倍数。

可能的正结果：compact/retrain 在部分漂移场景可摊销；逐层槽位复用限制峰值；K/logits 交接减少冗余读；在合适 GQA/span 形态下 CUDA 或固定分流改善端到端。可能的负结果同样需要保留：仅调 nprobe 足够、codes 小但 K gather 占主导、reader drain 阻碍复用、CXL 共同瓶颈使设备分流无用。

若 exact top-k 本身不满足质量，先改变预算/限定适用层，而不是继续优化 ANN。若 fixed centroid/固定维护已接近 oracle，停止复杂自适应控制；若成熟 executor 已接近最佳，停止自研 kernel；若调度/排序/merge 吃掉收益，退回 whole-group；若端到端主要受模型或 dense prefill 限制，报告边界并调整研究范围。收益依赖更改 mask、降质量或额外资源时，不能作为同语义系统加速结论。

## 7. 研究计划

采用依赖驱动的里程碑，工期在各里程碑测量后估算：

| 里程碑 | 交付与进入条件 |
|---|---|
| M0：框架与数值起点（已有） | 当前小模型生成、capture/replay、FP64/FlashInfer 数值记录；BF16 失败作为待定位问题保留 |
| M1：持续 session | 增量 turn、pending token、template prefix、position 与原生 full 控制一致；先短 context |
| M2：动态 IVF correctness | 单层/group，FlatIP、ID-only base+exact tail、安全 append 和 event trace；不先追求并发性能 |
| M3：替换与有界内存 | 二槽 pool、catch-up、整体 publication、真实 retirement、logical reset 和失败恢复；1M 可控 stream |
| M4：维护有效性 | 固定与自适应策略、topic/skew、真实 trace 与派生 multi-turn 质量；量化整个 session 摊销 |
| M5：高效率与内存层次 | native NUMA gather/CPU SIMD；实测热点决定 GPU kernel；接入实际 CXL placement 与 counters |
| M6：整模型与论文 artifact | 选定模型全层集成、长 context prefill 依赖、多轮 free generation、公平端到端图表与复现说明 |

M1/M2 是当前最直接的实现任务。M3 的环形复用不会替代 M1 的模型正确性，也不会替代 M5 的真实硬件证据。若 M4 无足够质量/成本空间，收敛研究问题后再投入 M5。

## 8. 参考文献与相关工作占位

以下已有定位用于限定贡献，完整 related-work 叙事和 BibTeX 在实验后补齐。论文 PDF 与来源笔记作为文献证据保留；owner 授权替换 research 提案与架构 HTML，并同步专题的当前研究定位。同步前记录的 wiki HEAD 为 `99cb2bf79e0a0e317c00b80e447c98e103bf2249`。历史输入散列、源码和运行版本见 [PROVENANCE.md](../../../cxl-kv-replay/docs/PROVENANCE.md)。

| 编号 / 来源 | 已有能力与本提案关系 | 后续需要补齐 |
|---|---|---|
| R1 [HAKES，PVLDB 2025](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)；[wiki](../../wiki/sources/hakes.md) | 并发 CPU IVF/PQ filter-refine；不是 CXL attention | 固定 artifact commit、参数和并发比较 |
| R2 [SIVF，HPDC 2026 / arXiv v3](https://arxiv.org/abs/2601.11808)；[wiki](../../wiki/sources/sivf.md) | GPU slab append/publish；不外推 host/CXL 性能 | 同硬件 GPU-resident 可运行范围 |
| R3 [InSituANN，arXiv:2608.08751v1](https://arxiv.org/html/2608.08751v1) | host IVF、紧凑过滤和 snapshot/delta 更新参照；通用 ANN 批查询 | 逐 token/小批适配；稳定 KV layout 与其 list-major 原向量的差异 |
| R4 [Quake，OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/mohoney) | adaptive partition maintenance；并发保证不能直接沿用 | 维护策略复现、版本/并发实现核对 |
| R5 [Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) / [metric](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) | add/search 需调用方同步；IP 不等于 cosine | 固定库版本、ID 和向量复制占用 |
| R6 [RetrievalAttention，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e36d4049fb0fea195a8267c8dcd0824-Abstract-Conference.html)；[wiki](../../wiki/sources/retrievalattention.md) | attention-aware graph、CPU/GPU partial 已有先例 | 固定上下文构建与持续 multi-turn 差异 |
| R7 [RetroInfer，PVLDB 2026](https://doi.org/10.14778/3796195.3796212)；[wiki](../../wiki/sources/retroinfer.md) | cluster index/buffer 联合设计、prefill/build 协作 | 不以“聚类索引+buffer”泛称新颖性 |
| R8 [FlashInfer，MLSys 2025](https://arxiv.org/abs/2501.01005)；[wiki](../../wiki/sources/flashinfer.md) | 成熟 paged attention、GQA 和 state merge | 锁定 backend 对 per-head mask/LSE 的支持 |
| R9 [Fluxion，2026 预印本](https://arxiv.org/abs/2605.07719)；[wiki](../../wiki/sources/fluxion.md) | hybrid sparse scheduling 已有先例 | 固定选集与 kernel/执行收益分别比较 |
| R10 [HiSparse，2026 预印本](https://arxiv.org/abs/2608.07009)；[wiki](../../wiki/sources/hisparse.md) | bounded GPU cache、exact sparse fetch | CPU partial 和 index lifecycle 的边界 |
| R11 [RULER 官方代码](https://github.com/NVIDIA/RULER) | 可控长度长上下文诊断 | 固定版本；派生 multi-turn 不冒充官方分数 |
| R12 [LongBench 官方代码](https://github.com/THUDM/LongBench) | 真实长上下文任务质量参照 | 固定数据/评分；分轮注入的适用性 |
| R13 [SAC，2026 预印本](https://arxiv.org/abs/2606.19746)；[wiki](../../wiki/sources/sac.md) | sparse selection + CXL KV 已有先例；native DSA/不同平台 | 分 track 比较，不宣称首次 CXL sparse attention |
| R14 [CXL-Vector owner 匿名稿](../../wiki/sources/cxl-vector.md) | DRAM codes、CXL 原向量 refine 已有先例 | 保持匿名/未确认录用状态，不外推至 IVF/multi-turn |
| R15 [CompactAttention](https://arxiv.org/abs/2605.16839)；[wiki](../../wiki/sources/compactattention.md) | prefill block union / paged access 邻近机制 | coverage 不等同于 per-head mask 等价 |

相关工作正文占位：动态向量索引的 reuse/rebuild；多轮 KV 存储与更新；异构 partial attention 和 kernel；CXL 容量与访问；memory lifecycle/reader reclamation。正式投稿前补齐各主题的系统性检索、出版状态、完整引用与实际 artifact 复现结果。
