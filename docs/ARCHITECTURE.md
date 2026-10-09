# 系统架构与数据契约

更新：2026-09-30。主线是多轮 session 的动态 IVF、partial attention 和有界内存管理；单节点 PCIe A100、CPU DRAM、memory-only CXL。研究动机和实验设计见 [PROPOSAL.md](PROPOSAL.md)，版本协议详见 [INDEX_LIFECYCLE.md](INDEX_LIFECYCLE.md)。

2026-10-09 平台划分更正：node 0/1 是统一 CPU/DRAM 资源池，node 2/3 是 memory-only CXL 资源池；不把两个 CPU socket 固定分成前台与后台。物理 NUMA locality 保留为分配/调度优化。当前内核只暴露 online nodes 0、1、3，node 2 尚不可测；具体计划与配置差异见 [plan.md](../plan.md)。

## 1. 架构总览与证据边界

![多轮 IVF 与 partial attention 的目标系统架构](figures/architecture.png)

图 1：全英文目标架构，非已测量系统。面板 (a) 为整体框架与分层内存，(b) 为逐层构建，(c) 为查询与 partial attention，(d) 为持续追加，(e) 为版本替换与回收。完整系统中的持续 session、IVF、CXL backing、native leases、CPU partial、混合路由和专用 GPU kernel 均待实现。图中成熟 attention 后端的集成已有数值证据，但不意味着整个数据流已运行。

V 在逐头选集确定后由执行路径获取；图中将这一步合并在执行规划与分层内存路径中。LSE 是内部
partial 归并与记录的状态，后续模型层消费 normalized output。

| 当前源码/证据 | 实际边界 | 目标扩展 |
|---|---|---|
| `module/interfaces.py` 的 ModelAdapter / TokenSelector / AttentionExecutor | 模型、选择、执行接口已分离 | session、索引版本和跨设备工作描述 |
| `module/generation.py` / `model.py` | 每次 generate 新建 DynamicCache，finally end_request；Llama-family、位置限制和 RoPE 范围明确 | 持续多轮、增量 prefill、最后一个 token 的缓存提交 |
| `module/cache.py` | 观察 append，借用模型拥有的 tensor；token_ids=None；不是稳定 native backing | 稳定 ID、K/V 目录、实际 placement 和异步租约 |
| `module/selectors.py` | full / exact-topk；Quest/retr-attn 为保留实验，后者 CPU NumPy exhaustive IP | IVF ID-only Flat 控制，SQ/PQ filter-refine，delta/rebuild |
| `module/executors.py` | SDPA、FlashInfer、FP64；在线返回 output，不提供多路 partial 合作 | 统一 partial 接口、CPU SIMD、可选 GPU logits+V / masked GQA |
| `kv_trace.py` / `reference.py` / `backends/flashinfer.py` | 单 step/group trace、数学 merge、自然对数 LSE adapter | 生命周期 trace、实际传输与性能计量 |
| [RESULTS.md](RESULTS.md) 和原始 runs | 小模型/合成数值检查；FP16 20/20、FP32 18/18；BF16 4 项 logits 失败 | 多轮质量、1M 索引持续性和 CXL 硬件实验 |

当前完整 KV 在模型设备上；无 CXL allocator、Faiss serving index、持续 session、稳定性能框架或优化 CPU/GPU 自研 kernel。当前 cache lease 的 `release(completion)` 仅接收参数并计数，不等待事件；现有 executors 没有返回实际 completion。未来分配器必须实现完整等待与回收，不能沿用当前计数逻辑冒充异步安全。

## 2. 组件、问题领域与实现映射

| 组件 | 要解决的问题 | 输入/输出与协作 | 拟议实现位置 |
|---|---|---|---|
| Session / 模型接入 | 多轮缓存重用、chat template 与 position 一致 | 新 turn token suffix → 真 Q/new KV；维护 committed/pending token | 扩展 `src/module/generation.py`、`model.py`、`interfaces.py` |
| KV owner / 目录 | 大历史容量、K/V 分开访问、稳定地址与回收 | append record → row ID、placement、KVLease；backing ready 事件 | 扩展 cache adapter；未来 `src/storage/` |
| IVF 检索 | 低成本插入、小批查询、Q/K 分布与召回 | Q + SnapshotHandle → 逐 head Candidates；不输出可直接用的近似 logits | 未来 `src/index/`；Faiss 控制和可替换自管 lists |
| 维护控制器 / 槽位池 | drift、skew、替换峰值、旧读者 | 历史统计 + credits → compact/retrain/replacement；发布 manifest | `src/index/`，详见 lifecycle |
| K 精排 / selection | 候选误差、避免 K 重读、保持逐 head 语义 | candidate union → canonical K 精确 QK → CSR + KLease + 可选 logits | 未来 `src/retrieval/`；首先 NumPy 控制 |
| 执行规划与 credits | 有限 DRAM/HBM、分散读、设备竞争 | 真实选集/地址 → resident、GPU packet 或 CPU partial 的 Work | 未来 `src/execution/`，先固定 whole-group |
| CPU/GPU executors | softmax/PV、传输与 launch 开销、GQA 复用 | Work + lease → normalized output + natural-log LSE + completion | 复用 `module/executors.py`；测得热点后才加 native/CUDA |
| 计量 / artifact | 防止逻辑 bytes、复制 bytes 和物理流量混淆 | identity、events、counters → 新 run 目录与阶段证据 | 扩展 `module/artifacts.py`、`scripts/` |

以上新目录是实现落点，不在文档修订中创建空 scaffold。通用接口不强制 import CUDA/Faiss；纯 NumPy trace 路径继续可用。首轮沿用 Transformers，避免重建 serving engine。

## 3. 持续 session 与逐层 prefill

目标接口为 `begin_session → append_turn → generate_turn → end_session`，不在 turn 结束时销毁 KV。完整 chat-template token 序列必须和 committed prefix 匹配，验证后只前向新增 suffix；不按文本字符长度切 suffix。若模板或 tokenizer 改变历史 prefix，显式拒绝或记录有成本的重算策略。position、causal mask、模型 revision 和 RoPE 配置在全 session 内一致。

现有 greedy loop 的最后一个 emitted token 尚未经过下一次 forward，因此其 KV 可能不在 cache。目标 session 需显式维护 pending token，在新 turn 前提交它以及所需 EOS/template 控制 token；不能漏掉最后一个 assistant token，也不能重复追加。

每层产生新 K/V 后发 append record，可启动该层的 assignment/encoding/offload。构建 scratch 可按层复用，但所有层/group 的活动索引必须常驻各自目录。索引工作与模型共享 CPU/GPU/链路资源，其时间计入首轮和新轮 TTFT。

prefill 仍保持 dense causal 语义；长历史下需要有界 chunked/streaming 实现与旧 KV 后备读取才能支持真实 1M 模型运行。现有 GPU DynamicCache prefill 没有实现这条路。1M index trace 压测与 1M end-to-end 是两个里程碑；稀疏 prefill 算法为后续独立研究。

## 4. 版本发布与索引生命周期

MVP 从单 session、单层、单 KV group、单 writer 开始。每个空间维护不可变 IVF base、可搜索增量区、精确 pending tail 和二槽替换池。snapshot handle pin 住 KV 目录、索引 generation、可见 token 前缀以及 delta 描述；index-ready 与 backing-ready 是不同事件。

新增条目先初始化、后发布。尚未入索引或尚未下放的合法 KV 由精确 tail 覆盖，包含当前 decode token。delta、mandatory 与 ANN 重复命中按逐 head token ID 去重。共同水位不得跨过未完成 hole。

构建 B 时 A 继续服务，新增写入记有界日志。B 必须追赶到明确 cutover 水位；参数改变则重新 assignment/encoding。短提交门内发布完整 `base + delta/tail + codebook + watermarks` manifest；读者安全 pin 与发布协同，不能分别交换几个裸指针。A 退役后等待 CPU/GPU/DMA/安装事件，才能复用。

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

基线冷数据路径是 `CPU 读 DRAM/CXL → 本地 pinned staging → PCIe H2D → GPU`，或 CPU 原地 partial 后只传状态。GPU direct/mapped CXL 访问仅为单独可选 backend，必须核对平台支持；架构不假设 A100 具有 CXL.mem 端口。

三层各自硬预算，计入 active/building/retired index、delta/log/pending tail、K/V 副本、candidate K、pinned memory、packet ring、partial、workspace、allocator reserved/slack。全局只准有限 layer/group 同时 rebuild；不要每层默认一份完整额外索引。4/8-byte IDs、32-byte code 和全量 Flat K 复制的容量差异详见 lifecycle。

credits 覆盖索引查询、K refine、V 获取和后台维护。必要读优先、预取用剩余 credits、维护分片且防止永久饥饿；维护影响 TTFT/ITL 与真实流量均计量。不要把进程整体绑定到无 CPU 的 CXL node；按 allocation 和 worker 定位，并采样页驻留。

## 7. 高效执行与 GPU kernel 的进入条件

先比较固定 whole-group CPU 与 GPU-pack、GPU hot + 固定 CPU history；只有真实选集、驻留与 span 的测量说明有净空间，才增加 group 内分流。每 head 的各 route 分区两两不交并恰好覆盖 S_h；resident、pending tail、ANN 历史的物理和逻辑交集必须去重。

- CPU 路径：读取原 K/复用 exact logits → FP32 stable softmax/PV → 小 partial state 回传。
- GPU K+V 路径：有界 gather/packet → 现成 paged attention → partial。先以当前独立 head 逻辑行作基线。
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
