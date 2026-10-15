# 系统架构与数据契约

更新：2026-10-15。主线是多轮 session 的动态 IVF、partial attention 和有界内存管理；单节点 PCIe A100、CPU DRAM、memory-only CXL。研究动机和实验设计见 [PROPOSAL.md](PROPOSAL.md)，版本协议详见 [INDEX_LIFECYCLE.md](INDEX_LIFECYCLE.md)。HAKES 源码已 vendor；独立 SIFT1M CPU run 已完成，项目 adapter 仍待接入。

**时序与主执行路径：** 本轮从已有大量历史（例如 100K）和活动 IVF 开始，轮内维护 delta 并定期插入活动索引，参数保持不变。全量重建只在下一轮 prefill 开始时启动，利用已保留历史 K 与新 suffix 产生的 K；初版在进入 decode 前完成追赶与发布。GPU 缓存有限热 KV（可含保留 prefix），GPU partial 与 CPU 历史 partial 合并输出。这是目标设计，尚未接入当前模型 runner。

2026-10-09 平台划分更正：node 0/1 是统一 CPU/DRAM 资源池，node 2/3 是 memory-only CXL 资源池；不把两个 CPU socket 固定分成前台与后台。物理 NUMA locality 保留为分配/调度优化。当前内核只暴露 online nodes 0、1、3，node 2 尚不可测；具体计划与配置差异见 [plan.md](../plan.md)。

## 1. 架构总览与证据边界

![多轮 IVF 与 partial attention 的目标系统架构](figures/architecture.png)

图 1：全英文目标架构，非已测量系统。面板 (a) 为整体框架与分层内存，(b) 为 prefill 阶段逐层构建，(c) 为 GPU/CPU partial attention，(d) 为轮内 delta 追加与定期入索引，(e) 为下一轮 prefill 启动的版本替换与回收。图中 packet 是可选搬运/缓存安装缓冲，不表示全部历史 K/V 都必须送 GPU。完整系统中的持续 session、在线 IVF/tail、模型 CXL backing、异步 native leases、CPU partial、混合路由和专用 GPU kernel 均待实现；本轮独立 index/extent infra 不等于完成这些路径。成熟 attention 后端已有数值证据，但不意味着整个数据流已运行。

V 在逐头选集确定后由执行路径获取；图中将这一步合并在执行规划与分层内存路径中。LSE 是内部
partial 归并与记录的状态，后续模型层消费 normalized output。

| 当前源码/证据 | 实际边界 | 目标扩展 |
|---|---|---|
| `infra/` 与 `configs/experiment.json` | 配置、预算 reservation、单 worker affinity、journal、只读 prepare 已提供；prepare 已运行 | 真实阶段计量、资源协调与自动 admission |
| `storage/arena.py` / `numa.cpp` | 严格 NUMA extent 源码已编译；没有执行 allocation/residency 验证 | 稳定 token 目录、模型 offload、GPU/DMA 完成事件和池化回收 |
| `index/interfaces.py` / `backends.py` / `faiss.py` | 通用候选协议与显式 backend/inventory；串行 Faiss 控制 API；HAKES 源码位于 `third_party/hakes`，独立 SIFT1M CPU 探索已运行；项目 adapter 未写、模型 runner 未调用 | HAKES 并发 filter、exact tail、模型 selector、Quake 维护对照与 bounded replacement |
| `module/interfaces.py` 的 ModelAdapter / TokenSelector / AttentionExecutor | 模型、选择、执行接口已分离 | session、索引版本和跨设备工作描述 |
| `module/generation.py` / `model.py` | 每次 generate 新建 DynamicCache，finally end_request；Llama-family、位置限制和 RoPE 范围明确 | 持续多轮、增量 prefill、最后一个 token 的缓存提交 |
| `module/cache.py` | 观察 append，借用模型拥有的 tensor；token_ids=None；不是稳定 native backing | 稳定 ID、K/V 目录、实际 placement 和异步租约 |
| `module/selectors.py` | full / exact-topk；Quest/retr-attn 为保留实验，后者 CPU NumPy exhaustive IP | IVF ID-only Flat 控制，SQ/PQ filter-refine，delta/rebuild |
| `module/executors.py` | SDPA、FlashInfer、FP64；在线返回 output，不提供多路 partial 合作 | 统一 partial 接口、CPU SIMD、可选 GPU logits+V / masked GQA |
| `kv_trace.py` / `reference.py` / `backends/flashinfer.py` | 单 step/group trace、数学 merge、自然对数 LSE adapter | 生命周期 trace、实际传输与性能计量 |
| [RESULTS.md](RESULTS.md) 和原始 runs | 小模型/合成数值检查；FP16 20/20、FP32 18/18；BF16 4 项 logits 失败 | 多轮质量、1M 索引持续性和 CXL 硬件实验 |

当前完整 KV 在模型设备上；基础 native extent 与 Faiss API 没有改变这条数据路径。
尚无 CXL 模型 offload、Faiss serving selector、持续 session、经验证性能框架或优化 CPU/GPU 自研
kernel。当前模型 cache lease 的 `release(completion)` 仅接收参数并计数，不等待事件；现有
executors 没有返回实际 completion。未来跨设备分配器必须实现完整等待与回收，不能沿用当前
计数逻辑冒充异步安全。

基础 allocation 的 owner 与 NumPy views 保留同一 resource reservation，源码意图是在最后一个
owner/view 消失后回收；该生命周期尚未运行验证，也不是 GPU/DMA completion 协议。native arena
按 extent 使用 `mmap/MPOL_BIND/first-touch/move_pages`，逐页查询失败或不匹配时拒绝分配。
Faiss 内部分配仍是未核验 NUMA 的普通 host memory；其 FP32 副本与显式 workspace/headroom
单独登记估算预算。配置的 tail/log/packet/version 上限目前是 P1–P3 实施参数，不表示已分配池。

当前主 IVF 源码选择 HAKES-Index，Faiss 保留串行控制，Quake 比较 adaptive maintenance。
配置/来源/能力分别记录于 `configs/experiment.json` 与 `index_backends.json`；选 HAKES
不会自动调用 Faiss。SIFT1M CPU run 是独立 ANN 证据，不等于模型 runner 已使用 HAKES。HAKES 的论文并发证据不等于本项目完成 runtime 接入：其 prefix predicate、
native adapter、allocator 和 generation retirement 都待实现。详细核对见 [plan §5](../plan.md#5-ivf-与相关系统的选型)。

## 2. 组件、问题领域与实现映射

| 组件 | 要解决的问题 | 输入/输出与协作 | 拟议实现位置 |
|---|---|---|---|
| Session / 模型接入 | 多轮缓存重用、chat template 与 position 一致 | 新 turn token suffix → 真 Q/new KV；维护 committed/pending token | 扩展 `src/module/generation.py`、`model.py`、`interfaces.py` |
| KV owner / 目录 | 大历史容量、K/V 分开访问、稳定地址与回收 | append record → row ID、placement、KVLease；backing ready 事件 | 扩展 cache adapter；未来 `src/storage/` |
| IVF 检索 | 低成本插入、小批查询、Q/K 分布与召回 | Q + SnapshotHandle → 逐 head Candidates；不输出可直接用的近似 logits | `src/index/` 通用协议；HAKES CPU 并发 filter 优先接入，Faiss 控制，Quake 维护对照 |
| 维护控制器 / 槽位池 | drift、skew、替换峰值、旧读者 | 轮内 flush；仅在下一轮 prefill 开始以历史统计 + credits 决定 rebuild；发布 manifest | `src/index/`，详见 lifecycle |
| K 精排 / selection | 候选误差、避免 K 重读、保持逐 head 语义 | candidate union → canonical K 精确 QK → CSR + KLease + 可选 logits | 未来 `src/retrieval/`；首先 NumPy 控制 |
| 执行规划与 credits | 有限 DRAM/HBM、分散读、设备竞争 | 逐 head 选集与 ready 热缓存 → GPU resident / CPU history 两个互斥 Work；packet 为搬运对照 | 未来 `src/execution/`，先固定热区与历史分工 |
| CPU/GPU executors | softmax/PV、传输与 launch 开销、GQA 复用 | Work + lease → normalized output + natural-log LSE + completion | 复用 `module/executors.py`；测得热点后才加 native/CUDA |
| 计量 / artifact | 防止逻辑 bytes、复制 bytes 和物理流量混淆 | identity、events、counters → 新 run 目录与阶段证据 | 扩展 `module/artifacts.py`、`scripts/` |

`infra/`、`storage/` 和独立 index API 已有实质源码；其余新目录仅是实现落点，不创建空
scaffold。通用接口不强制 import CUDA/Faiss；纯 NumPy trace 路径继续可用。首轮沿用 Transformers，
避免重建 serving engine。目录与执行状态见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 3. 持续 session 与逐层 prefill

目标接口为 `begin_session → append_turn → generate_turn → end_session`，不在 turn 结束时销毁 KV。完整 chat-template token 序列必须和 committed prefix 匹配，验证后只前向新增 suffix；不按文本字符长度切 suffix。若模板或 tokenizer 改变历史 prefix，显式拒绝或记录有成本的重算策略。position、causal mask、模型 revision 和 RoPE 配置在全 session 内一致。

现有 greedy loop 的最后一个 emitted token 尚未经过下一次 forward，因此其 KV 可能不在 cache。目标 session 需显式维护 pending token，在新 turn 前提交它以及所需 EOS/template 控制 token；不能漏掉最后一个 assistant token，也不能重复追加。

每层产生新 K/V 后发 append record，可启动该层的 assignment/encoding/offload。构建 scratch 可按层复用，但所有层/group 的活动索引必须常驻各自目录。索引工作与模型共享 CPU/GPU/链路资源，其时间计入首轮和新轮 TTFT。

后续 turn 的 prefill 开始是重建的唯一准入时点；已有约 100K 历史的 K 可立即供 CPU builder
扫描，不需要重新前向历史 token。GPU 按模型语义对新增 suffix 做 dense causal prefill，CPU
逐层/group 建新版本并追赶此时产生的新 K。初版在 prefill→decode 边界完成所需版本发布；
赶不上的构建计入 TTFT 等待，开始时预算不足则跳过重建并复用原版本，不把 full rebuild
移到本轮 decode。轮内 delta 达到批量或时间阈值只触发插入。

prefill 仍保持 dense causal 语义；长历史下需要有界 chunked/streaming 实现与旧 KV 后备读取才能支持真实 1M 模型运行。现有 GPU DynamicCache prefill 没有实现这条路。1M index trace 压测与 1M end-to-end 是两个里程碑；稀疏 prefill 算法为后续独立研究。

## 4. 版本发布与索引生命周期

MVP 从单 session、单层、单 KV group、单 writer 开始。Faiss 控制使用不可变 IVF base，目标
HAKES 后端可在同一 generation 内分区同步追加；两者都需精确 pending tail 和二槽替换池。
snapshot handle pin 住 KV 目录、索引 generation、可见 token 前缀以及增量描述；index-ready
与 backing-ready 是不同事件。HAKES 分区读写锁不提供本项目 prefix snapshot：scan 的 ID
可见性过滤必须早于候选 heap，不能仅事后删除 future IDs。

新增条目先初始化、后发布。尚未入索引或尚未下放的合法 KV 由精确 tail 覆盖，包含当前 decode token。delta、mandatory 与 ANN 重复命中按逐 head token ID 去重。共同水位不得跨过未完成 hole。

轮内 delta 按活动参数定期 flush：先初始化并发布 IVF 条目，再整体推进 indexed prefix 和 delta 描述；旧视图仍可 pin 其 delta，重叠 ID 逐 head 去重。下一轮 prefill 开始获准构建 B 时 A 保持有效，prefill 新写入记有界日志。B 必须在进入 decode 前追赶到明确 cutover 水位；参数改变则重新 assignment/encoding。短提交门内发布完整 `base + delta/tail + codebook + watermarks` manifest；读者安全 pin 与发布协同，不能分别交换几个裸指针。A 退役后等待实际借用 A 的任务完成，才能复用；独立 KV/缓存/传输 lease 各等自己的 CPU/GPU/DMA/安装事件。

槽位池不移动 canonical K/V，索引 rebuild 不使全部 KV 租约失效；索引 handle 与 KV lease 是不同 owner。活动/退役索引和异步 staging 都占 credits；没有空闲槽位时推迟维护或回压。状态机、追赶与 ABA 约束见 [INDEX_LIFECYCLE.md](INDEX_LIFECYCLE.md)。

## 5. 从向量检索到精确 attention

真实 post-RoPE Q/K 使用模型 scaled inner product；原 K 不归一化成 cosine。coarse assignment、查询 metric、residual encoding、dtype 必须记录。centroids/codes 的 contiguous lists 仅保证索引扫描局部性；ID-only Flat 与 exact refine 对 canonical token-major K 的 gather 可能仍零散。

对 head h，候选为 C_h，精排后历史选集为 T_h，最终 S_h=T_h∪mandatory_h。每 head 单独精排；候选 union 仅合并 K 物理读，最终 union 仅合并 K/V 读取。用 union 代替每 head 的 mask 会改变 attention 语义。

exact refine 留存选中 K 的 lease，可交给执行端避免第二次 K 读取。未选中候选及时释放；V 在最终选集确定后获取。exact logits 只有 request/session、step、head、layer/group、KV 版本、scale、position/score transform 和精度全部匹配才复用，禁止用 ANN/PQ 分数直接作为 logits。

```text
SnapshotHandle { session, layer, group, kv_layout_epoch, index_generation,
                 visible_hi, indexed_hi, base_hi, delta_refs, transform_id }
Candidates     { query_identity, head_indptr, token_ids, approximate_scores }
Selection      { query_identity, head_indptr, token_ids, mandatory_policy,
                 selected_K_lease, optional_exact_logits }
KLease         { owner, kv_layout_epoch, token_to_offset, precision, ready_event }
Work           { route, physical_spans, head_membership, buffer_slot, completion }
Partial        { query_identity, head, partition_id, output, natural_log_lse }
```

## 6. 内存放置、预算与协调

canonical 冷 K 与 V 分开存放于 CXL 层 `{2,3}`；活动 codes/centroids/list metadata 由统一 CPU/DRAM 层 `{0,1}` 管理。层内具体 placement 按 locality 和预算选择，GPU-local staging 是优化候选。HBM 留模型权重、模型 workspace、有限 hot KV、packets 与 partial states；CXL 是容量后备，不执行 ANN 或 attention。节点可用性必须核实，当前不得分配到未暴露的 node 2。已分配的 index slot 与 packet slot 分属不同 pool。

主路径由 CPU 读取选中的 DRAM/CXL 历史 K/V、原地计算 partial，再向 GPU 回传小状态；GPU
使用已驻留热 K/V 计算另一路 partial。热缓存是同一 session 的有限副本，可保留 prefix/sink
和过去访问的热点；缓存命中只改变执行位置，不改变既定 S_h，也不默认实现跨请求 prefix
sharing。缓存安装完成且 lease 被 pin 后才可纳入 GPU 路由；新生成 KV 在下放完成前保留
可读 owner。`CPU 读 → pinned staging → PCIe H2D → GPU packet` 保留为执行对照和热缓存
安装路径。GPU direct/mapped CXL 访问仅为单独可选 backend，必须核对平台支持；架构不假设
A100 具有 CXL.mem 端口。

三层各自硬预算，计入 active/building/retired index、delta/log/pending tail、K/V 副本、candidate K、pinned memory、packet ring、partial、workspace、allocator reserved/slack。全局只准有限 layer/group 同时 rebuild；不要每层默认一份完整额外索引。4/8-byte IDs、32-byte code 和全量 Flat K 复制的容量差异详见 lifecycle。

credits 覆盖索引查询、K refine、V 获取和后台维护。必要读优先、预取用剩余 credits、维护分片且防止永久饥饿；维护影响 TTFT/ITL 与真实流量均计量。不要把进程整体绑定到无 CPU 的 CXL node；按 allocation 和 worker 定位，并采样页驻留。

## 7. 高效执行与 GPU kernel 的进入条件

主设计是固定 GPU hot + CPU history partial，并以 whole-group CPU 与 GPU packet 作对照。
对 head h，pin 已 ready 的 GPU resident 集合 H 后，GPU 路由 `G_h=S_h∩H`，CPU 路由
`C_h=S_h\G_h`；新生成且尚未下放的 KV 保留在可读 GPU 路由。每 head 的各 route 分区
两两不交并恰好覆盖 S_h；同一缓存 token 仍可在 CPU 索引中命中，但不得再次在 CPU partial
贡献。物理 union 只合并读，不能改变 head membership。更复杂的动态分流根据测量后置。

- CPU 路径：读取原 K/复用 exact logits → FP32 stable softmax/PV → 小 partial state 回传。
- GPU 热 KV 路径：pin ready resident K/V → 现成 attention → partial；独立 head 逻辑行先保留正确 mask。有界 packet 是对照或 cache promotion 缓冲，不要求冷历史全量搬运。
- GPU logits+V 路径：复用真实 logits，只送选中 V 与描述；CUDA softmax/PV/merge kernel 是待实现候选。
- Masked GQA 路径：共享加载 KV tile，head membership 控制每个贡献；只有 per-head kernel 的重复读/launch 确为热点时才实现。

首先复用 FlashInfer 的成熟 attention，测量 plan/run、gather、H2D、tile/mask、merge 和同步；不预设 Tensor Core 比 SIMT 快。CPU 路径不使用 Python FP64 reference 作为性能基线。kernel 比较固定 mask、dtype、资源和 resident 状态；只测 kernel launch 不能支撑端到端加速。

异步 owner 在所有 completion 完成前保留 storage；generation 结束也要 drain。当前选部分 groups 会执行整层 SDPA 控制以保留其余 heads，造成重复计算，不能带着此诊断开销声称已优化模型。

## 8. 已实现的 Trace v1

文件由 `numpy.savez` 写入，不使用 pickle。输入仅接受可信的本地研究文件。

| 字段 | 形状 / 类型 | 语义 |
|---|---|---|
| q | [r,d] float32/float64 | 当前 step 的真实位置变换后 Q |
| k / v | [N,d] float32/float64 | 一个共享 KV head 的 canonical 内容 |
| indptr | [r+1] integer | head 的 CSR 边界 |
| indices | [sum_h |S_h|] integer | canonical K/V 的 row ID，非 position 或 packet offset |
| positions | [N] integer | 各 row 对应的原始 token position |
| metadata_json | scalar Unicode JSON | 下列元数据 |

必需元数据：`format_version=1`、`origin`、`model_id`、`model_revision`、`request_id`、`step`、`layer`、`kv_group`、`epoch`、`query_position`、`sm_scale`、`score_semantics=scaled_dot_product_post_rope`、`position_transform`、`source_dtype`。

真实模型额外记录 `capture_mode`、`prompt_id`、`q_head_ids`、`model_config_sha256`。`position_transform` 推荐为包含 RoPE 配置的 JSON object；代码只能验证存在，模型 capture 负责正确性。

BF16/FP16 tensor 可以无损转换成 float32 存入 NPZ。NPZ 的磁盘/host 占用因此不同于 canonical BF16 payload，两者不得混算。GPU adapter 检查目标 dtype 的往返转换是否保留数值，拒绝静默降低精度。

`GroupTrace` 的 frozen dataclass 只禁止重绑字段，不保证内部数组/字典物理只读。调用者负责把已记录 trace 当不可变证据。当前不是并发 runtime 的生命周期实现。

## 9. Partial state 数值契约

对于一个 head 的分区 P，s_i = scale · qᵀk_i：

```text
m = max(s_i)
z = sum exp(s_i - m)
o = sum exp(s_i - m) V_i / z
L = m + ln(z)
```

标准输出为 `(o, L)`，L 使用自然对数。空分区输出零向量和 `L=-inf`。

归并 A/B 时：`L=logaddexp(L_A,L_B)`，`o=exp(L_A-L)o_A+exp(L_B-L)o_B`；两个分区都空时单独返回空状态，避免 `-inf - -inf`。代码不持有 partition provenance，调用者必须保证不重不漏。

所查 FlashInfer 常规 attention/cascade 路径采用 base-2 LSE。当前 adapter 将它乘 `ln(2)` 转为内部格式；调用 FlashInfer merge 时应反向除以 `ln(2)`。更换 backend 时重新核实数值约定。

## 10. 已实现的 GPU 基线映射

- 每个非空 Q head 作为一个逻辑 request，`num_qo_heads=num_kv_heads=1`。
- Q 为 `[active_heads,1,d]`；K/V pool 为 `[unique_tokens,1,1,d]`。
- CSR indices 可以跨 heads 指向同一个 pool entry，保留存储去重。
- `page_size=1` 是 GPU adapter 视图；未来 CXL backing page 可以独立设为 16/32/64 tokens。
- Q/K 已做位置变换，kernel 使用 `pos_encoding_mode=NONE`。
- 独立行可能丢失 kernel 内 GQA 复用；后续以 membership/signature 分组测量该代价。
- 空 heads 不提交给 wrapper，结果保留零输出和 `-inf`。
