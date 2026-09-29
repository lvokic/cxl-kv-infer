# 研究依据与输入版本

交付日期：2026-09-29。研究建议和待实现架构不等于已测性能。

## 本地输入

知识库：`~/kvcache-wiki`，已提交版本 `b6b24fbc00fdc8febcf198a38a3a131036a1761a`。
交付时 `raw/research/` 是该知识库的未跟踪目录，因此下面单独记录输入散列，不能用 wiki commit 代替这些文件的版本。

这些相对链接用于两个仓库同在 `~/` 下的本地导航；把新仓库单独复制到其他机器后，需要另行提供研究材料。

| 输入 | SHA-256 |
|---|---|
| [Proposal v2.1](../../kvcache-wiki/raw/research/CXL_KV_Research_Proposal_2026-09-28.md) | `840f2e5f45531da6835e27447960e84890ca4a01a91a10f9403abe20485f12a4` |
| [Architecture v2](../../kvcache-wiki/raw/research/CXL_KV_Architecture_v2.html) | `69767d3cbb75446df8f9c6e8df24dc5dcaf808cfa2bf9e8e8b164fcd0c157338` |
| [Kernel Efficiency Brief](../../kvcache-wiki/raw/research/KV_Cache_Kernel_Efficiency_Research_Brief.html) | `a81f3d39e5706bd8ca14754f6436474b97ede9caed4584e8f04e6d350266227e` |
| [CXL-Vector manuscript](../../kvcache-wiki/raw/papers/SIGMOD27_CXL_Vector_P1297.pdf) | `e147525b7448e12f0d40f303c1b3feb5efe50a78b0d79edbaaf4b06567f4e4a7` |

原始材料保留在知识库，没有修改或复制到新仓库。CXL-Vector 是 owner 提供的匿名 SIGMOD '27 投稿稿；此记录不表示论文已录用。其源码尚未提供，当前复用判断依据稿件中的机制。

## 研究定位

- [CXL-Vector note](../../kvcache-wiki/wiki/sources/cxl-vector.md)：DRAM graph/codes、CXL 原向量精排；attention 的 V、GQA 和逐 head mask 是新增适配问题。
- [RetrievalAttention](../../kvcache-wiki/wiki/sources/retrievalattention.md) 与 [Fluxion](../../kvcache-wiki/wiki/sources/fluxion.md)：CPU/GPU 合作、partial attention 和异构任务调度已有先例。
- [FlashInfer](../../kvcache-wiki/wiki/sources/flashinfer.md) 与 [CompactAttention](../../kvcache-wiki/wiki/sources/compactattention.md)：稀疏访问、GQA 复用、page table 和 state merge 需要作为强基线。
- [RetroInfer](../../kvcache-wiki/wiki/sources/retroinfer.md) 与 [Quest](../../kvcache-wiki/wiki/sources/quest.md)：聚类/页摘要路线是图索引的重要对照。
- [Self-Indexing KVCache](../../kvcache-wiki/wiki/sources/self-indexing-kvcache.md)：压缩表示本身可参与选择，值得作为后续检索分支。
- [综合页](../../kvcache-wiki/wiki/topics/sparse-attention-cxl-anns.md)：文献之间的关系与现有证据边界。

研究假设是：在相同逐 head 选集和精度下，利用已经获得的 K / exact logits，以及物理驻留、分散度和共享关系，可以降低数据读取、传输和计算的总成本。它还需要真实 trace、相同资源的强基线、端到端时序以及模型质量评估。

`logits + V` 的 GPU 路径是对 proposal 已有 exact-logit 复用接口的优先实验化，不作为已证实的新颖性主张。独立 head 的 FlashInfer adapter 也是接口基线，尚未证明 kernel 效率。

所引用 API 的固定源码版本及具体入口见 [ENVIRONMENT.md](ENVIRONMENT.md)。
