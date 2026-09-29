# 代码起始交付文档

日期：2026-09-29。目标平台：双路 Xeon 6530P、A100 PCIe 40GB、256 GiB memory-only CXL。

## 1. 起始点与第一项决定

从 **固定逐 head 选集的单 group / 单层 replay** 开始。控制层采用 Python / NumPy；GPU 常规 attention 使用 FlashInfer；NUMA allocation、CPU gather / partial 逐步用 C++ 实现；Faiss 提供后续检索对照。

第一轮要回答：候选 K 已完成精排后，复用 K 和真实 logits 能否改变 CPU/GPU 的最优执行路径？

保留三条候选路径：

1. selected K + V → GPU QK / softmax / PV。
2. selected V + exact logits → GPU softmax / PV。
3. selected V + exact logits → CPU softmax / PV → 小型 partial state 回 GPU。

先固定 S_h 验证执行收益。改变候选 R、索引结构或最终 mask 的实验归入独立的检索/质量矩阵。

## 2. 本次交付了什么

| 文件 | 已编写内容 | 当前验证状态 |
|---|---|---|
| `src/cxl_kv_replay/trace.py` | 一个 KV group 的 trace、CSR 检查、NPZ 读写、小合成样本 | 未执行 |
| `src/cxl_kv_replay/reference.py` | FP64 selected attention、空分区、稳定 partial merge | 未执行 |
| `src/cxl_kv_replay/backends/flashinfer.py` | selected union payload + 独立 head 逻辑行；base-2 → natural-log LSE | 未执行；依赖未安装 |
| `src/cxl_kv_replay/cli.py` | 生成、查看、参考/FlashInfer replay；记录输入散列；独占输出 | 未执行 |
| `scripts/inspect_platform.py` | 只读 CPU / NUMA / GPU / toolkit 快照 | 未执行 |
| `configs/platform.json` | 本次会话观察到的硬件和建议放置 | 静态配置，不控制 allocator |
| `native/README.md` | 下一个 native 模块的接口与资源所有权要求 | 设计文档，无 native 实现 |

没有安装软件、占用 GPU 做实验或生成性能结果。当前代码不是完整的 CXL runtime；写在本文件中的验收要求是后续工作，并非已通过的检查。

## 3. 交给下一位实现者的任务清单

### T00：建立可重复环境和数值基准

**输入：** 此仓库、平台记录、独立虚拟环境。

**工作：** 确认 Python / CUDA toolkit / PyTorch / FlashInfer 的兼容组合；安装 core 依赖；执行小规模参考流程；在 owner 请求验证后开展数值检查；保存实际版本。GPU adapter 的源码审查 commit 不是已测试的安装版本。

**完成标准：** 同一 trace 的完整参考计算与不交分区 merge 一致；空 head、空分区和重叠 heads 有明确处理；GPU 与参考的精度差异有记录。先使用相同输入数值，禁止静默 BF16 转换改变比较对象。FP64 参考的容差与 BF16 GPU 的容差分别定义，不许为了通过而任意放宽。

**产物：** 实际环境清单、验证范围与结果；在完成前保持“未验证”标记。

### T01：采集一个真实 KV group，再扩成一层

**输入：** owner 选定并可访问的 GQA 模型。建议从可容纳于 A100 的 8B 级模型开始，checkpoint / revision 待明确。

**工作：** 在真实位置变换之后捕获 Q/K，保存 V；使用模型实际 `sm_scale`。记录 source dtype、RoPE 配置、request/step/layer/group、原始 token positions 与 head 映射。首次冻结一份逐 head exact-top-k 或既有 selector 的 mask，mandatory sink/window/delta 合并并精确去重。

**完成标准：** 加载文件能追溯到模型 revision 和配置；不同位置语义、score bias/soft-cap 等不受 v1 支持的模型明确拒绝；离线 teacher-forced capture 标注用途。未来 decode Q 不可进入在线索引构建。

**产物：** 小规模真实 trace、capture 脚本、位置与精度说明。Trace 不提交 Git。

### T02：NUMA store 和两个简单执行端

**输入：** 相同 trace；node 1 DRAM / node 3 CXL 两种 backing。

**工作：** 完成 native allocation / first-touch / residency 查询；K/V 分开；index、metadata、pinned staging 留在 node 1。实现 CPU historical partial，以及 CPU gather → pinned buffer → GPU packet direct attention。保留 GPU hot 独立分区。

**完成标准：** 实际页位置可证明；CPU 与 GPU 路径的逐 head 集合完全相同；内存预算有界；计入 Q D2H、descriptor 生成、gather、H2D、kernel、merge 和等待；引用生命周期覆盖全部异步消费者。

**产物：** 第一张“相同 mask 的 CPU / GPU 路径成本表”。此时不需要 ANN 或在线 planner。

### T03：让候选精排结果进入执行路径

**输入：** 已记录的逐 head 候选集合 C_h；真实执行候选 K 读取、原精度精排，产生最终 S_h。

**工作：** 实现 KLease + exact-logit provenance；比较保留/丢弃 lease；CPU 复用 logits；实现独立的 GPU `logits + V` softmax/PV operator，与现有 FlashInfer `K + V` 路径比较。FlashInfer adapter 本身不提供“输入预计算 logits”的此类接口。

**完成标准：** 不把离线 logits 当成免费在线输入；C_h 内独立 top-k，不跨 head 扩充候选；计入打包、metadata、两侧中间状态及同步；KV / logits 的 dtype、head、step、epoch 全部匹配。所有路径共享同一 CXL 资源预算。

**产物：** `K+V GPU` / `logits+V GPU` / `logits+V CPU` 的相同选择结果对照，以及收益发生条件。

### T04：接入 Faiss，再决定 RoarGraph 投入

**工作：** IndexFlatIP → IVF-Flat → IVF-SQ8 + 共同精排。显式使用 inner product；原 K 不做 cosine normalization。记录 residual encoding、nlist/nprobe、候选 R、index 内存及完整构建成本。

**完成标准：** exact top-k、近似选择和 full attention 的误差分开；质量校准与评估数据分开；Faiss 内部 FP32 副本计入内存，不能当成外置 BF16 canonical K 的零拷贝索引。

RoarGraph 在压缩 IVF / 页摘要基线建立后接入。使用相同存储与精排后端，对照原精度导航、SQ8 导航；CXL-Vector 源码位置目前未知，尚不能指定复用模块。

### T05：有证据后增加 planner 与完整模型

只有 whole-group / 静态路线与可实现的组内分流之间存在稳定净空间，才实现 planner。初始最多一个 GPU bulk + 一个 CPU residual；先计算分流成本，再增加负载反馈。

完整模型优先尝试 RetroInfer 的 `model_hub/LLM.py::layer_decode`，让本仓库后端替换 attention 路径。其原生 selection / estimation 作为算法对照，不能混入“固定选集”的执行对照。SGLang serving、并发 graph mutation、PP 放在这一阶段之后。

## 4. 第一轮收敛标准

值得继续的信号：

- 同一 mask、相同资源下，数据状态 / GQA overlap / 物理分散度导致路径胜负发生可解释变化。
- KLease 或 exact logits 复用的收益大于保持中间状态、打包和归并成本。
- 相同任务质量下，索引路线减少真正读取的候选 K，而不仅是返回 token 数更少。

收缩设计的信号：最佳简单路径已接近离线 oracle；索引或权重/MLP 占主导；所谓收益依赖更多 DRAM/CPU 或更低质量；CXL 在容量限制下没有体现独立价值。

研究产物优先是三张图：路径相图；候选/选中数量到真实流量和时间的分解；whole-group 与组内分流的净收益。proposal 的约 15–20% 内部投入门槛是研究管理参考，不是预期结果或投稿保证。

## 5. 当前仍需在后续实现中落实的输入

- 模型 checkpoint / revision 和可用的真实输入数据。
- CXL-Vector 源码路径或仓库，以确定可复用模块和编译方式。
- 实际 toolkit / Python 依赖组合，以及无人竞争的 GPU 实验时段。
- 页位置查询和硬件 CXL/IMC/PCIe counters 在此机器上的可用性；没有 counters 时分别报告逻辑字节和估计值。

这些输入不妨碍从当前 trace 和接口骨架开始工作。
