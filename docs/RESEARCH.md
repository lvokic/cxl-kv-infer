# 研究计划与框架对照

更新：2026-09-29。研究材料与论文出处见 [PROVENANCE.md](PROVENANCE.md)，实验数值见
[RESULTS.md](RESULTS.md)。本页记录决策和待验证任务，不把计划写成实验结论。

## 问题和边界

主问题：在单节点 A100、CPU DRAM 与 memory-only CXL 上，固定逐 Q-head 选集和精度后，
能否利用精排阶段已经获得的 K / exact logits，以及 KV 的真实驻留和物理分布，降低
sparse decode 从 Q-ready 到 output-ready 的总成本？

检索与执行是两条实验轴。执行实验固定 Q/K/V、逐 head 选集、精度和资源，只改变读取、
传输和计算路径。检索实验改变索引和候选集合，并单独测模型质量。离线 oracle、近似 ANN
分数和预先算好的 logits 不能算作在线免费输入。

保持这些约束：

- 每个 Q head 使用自己的最终选集；union 只能减少物理读取，不扩充 attention mask。
- 每个 head-token 贡献恰好执行一次；位置变换、score、精度和 request/layer/group/epoch 一致。
- approximate ANN score 不能当作 exact attention logits。
- K 与 V 分别计量；K 副本、H2D、索引副本、workspace 和 buffer 都计入容量与时间。
- 只有验证页实际落在 node 3，才可以称为 CXL backing；逻辑 payload 不等于硬件流量。
- teacher-forced trace 用于机制诊断；模型质量必须用独立的自由生成或任务评估。

## QUEST 和 RetrievalAttention 源码结论

两份代码都覆盖完整模型层、KV cache 生命周期和 attention 执行，因此模型接入、cache API、
位置编码与执行 kernel 之间的**软件边界可以复用**。本仓库现已提供统一接口：模型 adapter
返回 logits/cache，cache lifecycle 管理 append 与只读 lease，selector 输出独立逐 head row IDs，
executor 执行给定逻辑选集。当前 Transformers DynamicCache adapter 是 HBM/CPU 观测实现；
QUEST/RetroInfer 的 selector 语义和 cache 物理布局仍须作为独立 adapter 评估。

| 层 | QUEST | RetrievalAttention / RetroInfer | 对本仓库的启发 |
|---|---|---|---|
| 模型入口 | `quest/models/QuestAttention.py` 自行实现 Llama attention | `model_hub/LLM.py::layer_decode` 明确展开 norm→QKV→位置变换→cache 更新→attention→projection→MLP | 已实现 `ModelAdapter` / `ModelStep` 与共享 greedy loop；模型/RoPE 适配和执行端保持分开 |
| Cache owner | `quest/utils/kv_cache.py::KvPool/KvCache` 分配 GPU 上 `[layers, capacity, K/V, page, heads, dim]` pool，维护 block indices；`InferenceController` 管页面预算和请求元数据 | `cache_hub/cache.py::KV_Cache` 定义基类；`retroinfer_cache_gpu.py` 管 GPU/CPU KV、cluster 元数据、steady/retrieval/estimation zones 和 WaveBuffer | 已实现 `KVCacheLifecycle` / `KVCacheLease`；当前 Transformers adapter 观察原生 DynamicCache 并发只读 lease，不做复制或迁移。新 location、页表、索引和 budget 仍是 adapter 的职责 |
| Selector | 跟踪每页 K 的最小/最大值，`QuestAttention.forward` decode 时据当前 Q 估计 page score 并 top-k；prefill 走独立 kernel | RetroInfer 按 cluster 检索高价值区，以估计区近似注意力贡献，并让 GPU/CPU buffer 与 attention 执行共同调度 | 把 `Selector` 的候选/选集输出做成显式逐 head CSR；不要把 QUEST page budget 或 RetroInfer estimator 隐藏进 cache adapter |
| Kernel | 自己的 page append/prefill/decode/select CUDA/PyBind 算子；paged layout 与 QUEST 算法紧密相连 | 自定义 gather/reorganization、GPU attention 和 weighted partial merge，并复用 FlashAttention 路径 | 已实现 `AttentionRequest` / `AttentionExecutor`，接入 SDPA、FlashInfer 与 FP64 reference；后续 kernel 消费显式 route/indices/leases/precision contract |
| 目标系统 | GPU paged KV 上减少逐步扫描的 KV pages | CPU/GPU 异构 KV、聚类检索、offload/refill 和估计注意力 | 本项目另测相同选集下 CPU DRAM / CXL / GPU 的执行成本，再单独接入索引 |

证据来自本机 checkout 的源码：QUEST `Quest/quest/models/QuestAttention.py`、
`Quest/quest/utils/{kv_cache,controller}.py`、`Quest/kernels/include/{decode,topk}/`；
RetroInfer `RetrievalAttention/model_hub/LLM.py`、`cache_hub/retroinfer_cache*.py`、
`attn_hub/retroinfer_attn.py`。固定 commit 和工作区说明见 [PROVENANCE.md](PROVENANCE.md)。

两项目的依赖也不能直接共享：QUEST 声明 PyTorch 2.5.0 / Transformers 4.45.2；
RetroInfer requirements 声明 PyTorch 2.5.1 / vLLM 0.6.5 / Transformers 4.49.0，并带自定义 C++/CUDA 扩展。
当前仓库在独立 Python 3.12 / PyTorch 2.14.0 CUDA 13.0 环境运行，因此保持只读参考，
逐条抽取接口和算法，不把它们的环境安装进本项目。

### MInference Quest / RetrievalAttention module 接入

本机 `~/MInference/minference/modules/quest.py` 的 `quest_decode_kernel` 使用 query 符号和每块
K 极值估计 chunk 上界，再取块并执行 attention；`retr_attn.py` 的 `RetrAttnCache` 将中间 K
建成 CPU `papyfaiss.FlatIndex`，保留首尾 GPU K/V，并用检索与保留区的 partial softmax/LSE 合并。
本项目在既有 `TokenSelector` 插件上实现对应的 `quest` 与 `retr-attn` selector，并保留当前
model/cache/executor 分层。`retr-attn` 暂以 NumPy exhaustive Flat-IP 作为可运行候选生成器，
记录 host copy bytes；GPU executor 再以原 K/V 计算 exact logits。这样数值语义遵守本项目的
attention contract，但性能路径尚不等价于 MInference 的 papyfaiss、异步 cache 或 FlashAttention
组合，也没有声称复现其吞吐。运行参数见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 实验顺序

1. **模型和质量：** 从小型集成模型扩展到研究用 GQA 模型；冻结 prompts、context、layer/step、
   精度和 selector；比较 full attention 与 sparse free generation 的任务质量。
2. **固定选集执行：** HBM-only 数值参照、GPU hot + CPU cold partial、GPU hot + packed cold KV。
   比较前所有路径复用同一 trace / 同一逐 head mask。
3. **精排复用：** 真正执行候选 K 冷读与原精度精排，再比较 K+V GPU、exact-logits+V GPU、
   exact-logits+V CPU。记录 lease、打包、搬运、计算和 merge。
4. **检索：** FlatIP → IVF-Flat → IVF-SQ8 + 共同精排；其后才引入 RoarGraph / QUEST 页摘要，
   每个检索器都单独报告 recall、选集差异、模型质量、index bytes 和 build/update cost。
5. **调度：** 先量出 whole-group 路径；只有 GPU bulk + CPU residual 扣掉 planner/join/merge 后
   留有稳定净收益，才实现在线 planner。

首批可测维度：32K / 64K / 128K 上下文（限模型位置上限）、历史 k=256/1024、候选 R/k=1/2/4、
node 1 DRAM / node 3 CXL / node 0 DRAM、少量 node 1 CPU workers、不同 HBM residency、真实
GQA mask overlap。逐次增加一个维度，避免扫完整笛卡尔积。

## 计量和停止条件

单层主要指标是包含排队与等待的 Q-ready → output-ready wall time。阶段拆分为 Q/selection、
candidate K gather/refine、selection + K/logit handoff、V gather/packet/H2D、attention 或
softmax-PV、partial merge。Setup、JIT、分配和 graph capture 单列；CUDA event 与 CPU monotonic
clock 分别使用，不直接拼接成统一时间轴。并发路径要在共享链路上测 contention。

同时记录 `sum_h |C_h|`、候选 union、`sum_h |S_h|`、选中 union、逻辑 K/V payload、H2D bytes、
descriptor/workspace/staging 和模型/KV/index 的峰值。硬件 counters 不可用时使用
`logical_payload_bytes` 或 `estimated_transaction_bytes`，不得标为 measured CXL bytes。
容量受限实验要明确 DRAM budget；仅将 DRAM 容得下的数据搬到 CXL 不足以证明容量价值。

如果 whole-group 已接近 oracle，或 planner/join/merge 吃掉组内分流收益，就停止增加调度复杂度。
如果候选 K 冷读占主导，优先改进检索；如果 MLP/权重占主导，先估算端到端可优化空间。
