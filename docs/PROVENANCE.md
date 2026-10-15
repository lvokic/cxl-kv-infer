# 研究依据与输入版本

更新：2026-10-12。来源用于核对设计和代码边界，不构成此仓库的性能证据。

## 2026-10-12 IVF 选型修订与静态源码核对

owner 要求主研究后端不限于 Faiss，优先考虑真实更新/查询并发。相邻 wiki 的
[并发 IVF 分析](../../kvcache-wiki/wiki/analysis/concurrent-ivf-read-write.md)、
[HAKES](../../kvcache-wiki/wiki/sources/hakes.md) 和
[SIVF](../../kvcache-wiki/wiki/sources/sivf.md) 作为只读输入，本次没有修改 wiki。

- 查阅官方 [Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)、
  [HAKES 论文](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)/
  [源码](https://github.com/nusdbsystem/HAKES-Search)、
  [Quake 论文 §8.2](https://www.usenix.org/system/files/osdi25-mohoney.pdf)/
  [官方开发指南](https://marius-project.org/quake/development_guide)、
  [SIVF v3](https://arxiv.org/abs/2601.11808v3)/[源码](https://github.com/hpdic/ElasticIVF)。
  优先接入 HAKES CPU concurrent filter；Faiss 只保留串行控制；Quake 单列动态维护。
- `git ls-remote ... HEAD` 实际观察：HAKES
  `9fbe8b99371c70d854618cff6de60e9986f4bd95`；Quake
  `0651c88c5c71343191ef06ee4ed292cb4a01e1aa`；ElasticIVF
  `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`。与计划中已有 pin 一致，不用 mutable HEAD 运行。
- HAKES 源包下载至 `artifacts/upstream/hakes-9fbe8b9.tar.gz`，2,378,560 bytes，SHA-256
  `8de74f1e40db97ed89b78604d26ceb8f7924eee2e19187fc2540849fc93a7414`；解压原件保留。
  核对了 HakesIndex、BlockInvertedListsL、IndexIVFFastScanL、index_io_ext、参数生成器与
  CMake，逐文件 hash 和静态发现保存在 [source-audit.json](../runs/ivf-selection-20261012-followup/source-audit.json)。
  分区锁、filter-only 与源码缺陷仅为静态发现，没有执行故障复现、构建或并发操作。
- 修改 `interfaces.py`/`backends.py`、配置/catalog、基础 preflight 的依赖 inventory 和文档。
  `index_threads` 代替 `faiss_threads`；`requirements/native.lock` 仅锁 pybind11，
  `ann.lock` 保留 stock-Faiss control。两锁重新解析成功，已有环境未安装/卸载/升级任何包。
- 本次仅做源码阅读、lock 解析、文档/配置/代码修改与 whitespace diff 检查；
  **没有执行修改后的 adapter/config/prepare，没有编译 HAKES，没有测试或 benchmark。**
  旧 infra prepare 产物仍保存原始 Faiss 配置，不应当作当前 HAKES 可运行的证据。

研究源、用途和能力声明见 `configs/index_backends.json`，具体接入问题见
[plan §5.2](../plan.md#52-hakes-接入顺序与已经发现的边界)。当前 HAKES adapter 未写，
显式选型只决定接入优先级，不能被报告为已完成 SOTA index 或已测 CXL 系统。

## 2026-10-12 基础 infra

本轮起始 HEAD 为 `487dee387079147ff6ea2803bbe1775e94689973`，工作区 clean。当前实现是
其上未提交修改；run 的 manifest/source.zip 保存 dirty 源码，含 Python/C++、CMake、
requirements 和配置，另记录 native binary hash。没有修改相邻 wiki 或既有实验输入/输出。

- 主模型：本地 `/mnt/nfs/home/huangyirui/data/models/Llama-3.1-8B-Instruct`；canonical
  `meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659`，从
  ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct@359efdbb8af05b788a4ad4185215c6b8caa9052c`
  下载。2026-10-09 完成逐文件原仓匹配与分片结构检查；本轮仅核对大小、metadata hash 和下载记录。
- 基础 infra 时新安装：Faiss CPU 1.15.1、pybind11 3.1.0。当时 `requirements/ann.lock` 保存解析版本/哈希，并
  约束现有 inference lock；旧 NumPy 2.5.3、Torch 2.14.0+cu130、Transformers 4.57.6、
  FlashInfer 0.7.0 保留。后来拆分 native/control locks，原始锁在原 run/source.zip 可追溯。
  不是上游 Faiss HEAD 的源码构建，也未将 wheel 版本冒充 Git commit。
- Native 构建：CMake 3.28.3、G++ 13.3.0、libnuma1/libnuma-dev
  `2.0.18-1ubuntu0.24.04.1`。C++17 Release，两个编译任务；flags/commands/binary SHA-256 在
  [构建记录](../runs/native-build-20261012-infra-retry/build.json) 与同目录 compile_commands。
  [首次失败](../runs/native-build-20261012-infra/build.log) 为 libnuma/Linux 头文件常量冲突，已修复并保留。
- [最终 DRAM 只读准备](../runs/infra-20261012-dram-final/readiness.json) 和
  [单 CXL 只读准备](../runs/infra-20261012-cxl3-readiness/readiness.json) 保存 source、dependency、
  topology、GPU 利用和本地 baseline commits。观察 online nodes 0、1、3；node 2 仍缺席，
  endpoint/decoder 映射和 counters 留给运维核实。
- RULER 目录为官方 commit `e8bbff677ca2c239640dc90f93310dcf32408c93`，README 已改用
  NeMo-Skills 工作流；没有生成任务数据或安装其 serving/generator 依赖。源码及 README hash
  记录于 run/benchmarks.json。本轮 GitHub API 历史查询返回 403，未据此编造 legacy generator commit。

实际执行仅为 dependency 安装、editable 注册、native 编译与只读预检。没有运行单元/数值/性能
测试、Faiss train/add/search、NUMA allocation/residency、8B 模型或任务质量评估。候选方法与
拟议验收见 [DEVELOPMENT.md](DEVELOPMENT.md)，不能以这批源码或编译产物声称 CXL 吞吐/驻留已测。

## 本地输入

知识库：`~/kvcache-wiki`，2026-09-29 交付检查时的已提交版本为 `b6b24fbc00fdc8febcf198a38a3a131036a1761a`。
交付时 `raw/research/` 是该知识库的未跟踪目录，因此下面单独记录输入散列，不能用 wiki commit 代替这些文件的版本。

这些相对链接用于两个仓库同在 `~/` 下的本地导航；把新仓库单独复制到其他机器后，需要另行提供研究材料。

| 输入 | SHA-256 |
|---|---|
| [Proposal v2.1](../../kvcache-wiki/raw/research/CXL_KV_Research_Proposal_2026-09-28.md) | `840f2e5f45531da6835e27447960e84890ca4a01a91a10f9403abe20485f12a4` |
| [Architecture v2](../../kvcache-wiki/raw/research/CXL_KV_Architecture_v2.html) | `69767d3cbb75446df8f9c6e8df24dc5dcaf808cfa2bf9e8e8b164fcd0c157338` |
| [Kernel Efficiency Brief](../../kvcache-wiki/raw/research/KV_Cache_Kernel_Efficiency_Research_Brief.html) | `a81f3d39e5706bd8ca14754f6436474b97ede9caed4584e8f04e6d350266227e` |
| [CXL-Vector manuscript](../../kvcache-wiki/raw/papers/SIGMOD27_CXL_Vector_P1297.pdf) | `e147525b7448e12f0d40f303c1b3feb5efe50a78b0d79edbaaf4b06567f4e4a7` |

上表是 2026-09-29 输入的历史散列，不代表 wiki research 文件的当前内容。2026-09-30 owner 随后明确要求替换 research proposal 与 architecture HTML，已同步当前修订稿，详见文末记录；论文与 kernel brief 保留。CXL-Vector 是 owner 提供的匿名 SIGMOD '27 投稿稿；此记录不表示论文已录用。其源码尚未提供，当前复用判断依据稿件中的机制。

## 上游代码检查

本机只读代码检查锁定到下列 checkout：

| 项目 | 本机目录 | HEAD | 检查入口 |
|---|---|---|---|
| QUEST | `~/Quest` | `01c1623bf9395009520874e989e29f683203b357` | `quest/models/QuestAttention.py`; `quest/utils/{kv_cache,controller}.py`; `kernels/include/{decode,topk}/` |
| RetrievalAttention / RetroInfer | `~/RetrievalAttention` | `03f912c6e917c380d9d90c5ec85bb0f161ba53ef` | `model_hub/LLM.py::layer_decode`; `cache_hub/retroinfer_cache*.py`; `attn_hub/retroinfer_attn.py` |
| MInference | `~/MInference` | `29ef1974fcb5a1440ccf92d7d7996900230fddcf` | `minference/modules/quest.py::quest_decode_kernel`; `minference/modules/retr_attn.py::{RetrAttnCache,retr_attn}` |

QUEST checkout 有未提交 CMake 修改及额外实验文件；检查期间没有修改它。上表只用于标识 HEAD，
不声称所有本地文件都处于 clean 状态。RetrievalAttention checkout 当时为 clean。
上游接入检查时 wiki commit 为 `b6b24fbc00fdc8febcf198a38a3a131036a1761a`，该阶段未修改 wiki；后续 research 同步另见文末记录。
MInference checkout 在接入检查时工作区 clean；其 MIT license 保留于上游。当前仓库仅实现了
本地 selector adapter，没有复制上游实现文件或引入其依赖。

源码观察和共用边界见 [RESEARCH.md](RESEARCH.md)。上游依赖版本与本仓库环境不同，详见
[DEVELOPMENT.md](DEVELOPMENT.md)；没有安装上游依赖或运行上游 benchmark。

## 研究定位

- [CXL-Vector note](../../kvcache-wiki/wiki/sources/cxl-vector.md)：DRAM graph/codes、CXL 原向量精排；attention 的 V、GQA 和逐 head mask 是新增适配问题。
- [RetrievalAttention](../../kvcache-wiki/wiki/sources/retrievalattention.md) 与 [Fluxion](../../kvcache-wiki/wiki/sources/fluxion.md)：CPU/GPU 合作、partial attention 和异构任务调度已有先例。
- [FlashInfer](../../kvcache-wiki/wiki/sources/flashinfer.md) 与 [CompactAttention](../../kvcache-wiki/wiki/sources/compactattention.md)：稀疏访问、GQA 复用、page table 和 state merge 需要作为强基线。
- [RetroInfer](../../kvcache-wiki/wiki/sources/retroinfer.md) 与 [Quest](../../kvcache-wiki/wiki/sources/quest.md)：聚类/页摘要提供选择与 buffer 协作的既有参照；本阶段持续 IVF 主线不继续调优这两个 adapter。
- [Self-Indexing KVCache](../../kvcache-wiki/wiki/sources/self-indexing-kvcache.md)：压缩表示本身可参与选择，值得作为后续检索分支。
- [综合页](../../kvcache-wiki/wiki/topics/sparse-attention-cxl-anns.md)：文献之间的关系与现有证据边界。

当前研究主线是多轮持续 IVF 的低成本追加、复用/替换和有界内存管理，并在固定逐 head 选集下研究 K / exact logits 交接与执行效率。它还需要真实多轮 trace、生命周期测量、相同资源的强基线、端到端时序和模型质量评估；早期图索引路线保留为历史依据。

`logits + V` 的 GPU 路径是对 proposal 已有 exact-logit 复用接口的优先实验化，不作为已证实的新颖性主张。独立 head 的 FlashInfer adapter 也是接口基线，尚未证明 kernel 效率。

当前安装组件、固定版本和重建命令见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 2026-09-30 中文 proposal / architecture 修订

首次只读核对现有源码、原始 run 的 summary/checks/manifest 与 wiki 后，生成仓库内修订稿 [PROPOSAL.md](PROPOSAL.md)，重写 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [INDEX_LIFECYCLE.md](INDEX_LIFECYCLE.md)。该阶段未修改原始材料；随后 owner 明确授权的 research 同步见文末。

- 本仓库检查时 HEAD：`60e69caa3d5aff645dae29eec938ab36df86ba64`；本次是其上未提交的文档变化，旧 runs 对应版本以各自 manifest/source.zip 为准。
- wiki 检查时 HEAD：`99cb2bf79e0a0e317c00b80e447c98e103bf2249`。当时三份研究输入重新核算的 SHA-256 与本页历史表一致；随后同步前的 `git ls-files` 确认三份 research 文件已被跟踪，旧稿可由 wiki Git 历史追溯。
- 本地重点证据：[并发 IVF](../../kvcache-wiki/wiki/analysis/concurrent-ivf-read-write.md)、[HAKES](../../kvcache-wiki/wiki/sources/hakes.md)、[SIVF](../../kvcache-wiki/wiki/sources/sivf.md)、[RetroInfer](../../kvcache-wiki/wiki/sources/retroinfer.md)、[Fluxion](../../kvcache-wiki/wiki/sources/fluxion.md)、[FlashInfer](../../kvcache-wiki/wiki/sources/flashinfer.md)、[CXL-Vector](../../kvcache-wiki/wiki/sources/cxl-vector.md)、[HiSparse](../../kvcache-wiki/wiki/sources/hisparse.md)、[综合页](../../kvcache-wiki/wiki/topics/sparse-attention-cxl-anns.md)。
- 网络核验补充：[InSituANN arXiv:2608.08751v1](https://arxiv.org/html/2608.08751v1)，2026-08-09 的预印本，§3–4 与附录 A；[Quake OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/mohoney)；[Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) 与 [metric](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)；[HAKES 官方 artifact](https://github.com/nusdbsystem/HAKES-Search)；[SIVF arXiv](https://arxiv.org/abs/2601.11808)；[RULER](https://github.com/NVIDIA/RULER) 和 [LongBench](https://github.com/THUDM/LongBench)。网页/代码主分支是当日检查依据，未冒充固定 artifact commit；运行上游前另锁 commit。
- 既有运行的依赖记录：Python 3.12.14、NumPy 2.5.3、PyTorch 2.14.0+cu130、Transformers 4.57.6、FlashInfer 0.7.0；来自 environment imports 和真实模型 manifest，本次未重新安装/运行。
- 既有 CPU numerical summary 为 482 项，GPU summary/checks 为 644 项合计；修正 RESULTS 中把总数重复归到每种 dtype 的表述，没有修改 raw checks 或重新实验。

### 架构图 artifact

使用用户指定的 [imagegen 技能](/home/huangyirui/.codex/skills/.system/imagegen/SKILL.md) 的内置 image_gen 工具生成，未使用 CLI/API fallback。图中表示目标架构，非已实现系统。

2026-09-30 按用户追加要求重绘为全英文：参考用户提供截图的分面组织、数据块、索引表和图例样式，形成 System Framework、Layer-wise Build、Query & Partial Attention、Incremental Append、Replace & Reclaim 五个面板。截图作为风格参考，不复制其中的图索引/SSD/删除语义；本图继续描述持续 IVF 和 memory-only CXL。

样式参考：[用户截图](../截屏2026-09-30%2012.30.57.png)，SHA-256：`d0426995ec0cafcb0895fc839ff06aeed023440f91740a679a43101c4b6dbb66`。截图留在原路径，未修改。人工检查后修正 CPU-mediated CXL I/O、K-only index encoding、全部 unindexed tail、parallel partial merge 和 retirement 箭头。

项目图片：[figures/architecture.png](figures/architecture.png)。完整生成与编辑 prompt：[figures/architecture.prompt.txt](figures/architecture.prompt.txt)。内置工具未提供可锁定的生成模型 revision，因而不虚构其版本；代码/论文实验版本另按上述记录处理。

最终图片为 1536×1024 RGB PNG，SHA-256：`551717a12acaaa2cedf1f5a0724dd377483adffe2560cca7953c179d21d0d688`。
prompt 文件 SHA-256：`6319aff2fafd7ed76589cae76c0c51717f3c3878ee747d4a843759ec102611bf`。

## 2026-09-30 owner 授权同步 wiki research

owner 明确要求进入 wiki、用现有 proposal 替换 `raw/research` 提案、放入架构图解释模块责任，并更新 `CXL_KV_Architecture_v2.html`。该授权覆盖本仓库和 wiki 对 raw 只读保留的默认规则，仅对指定研究材料执行替换；论文 PDF、source notes、kernel brief、既有 runs 与 trace 均保留。

同步前 wiki HEAD 为 `99cb2bf79e0a0e317c00b80e447c98e103bf2249`，工作区 clean；本次同步是未提交变化。保留研究文件的旧名称，正文更新为 2026-09-30：

| 当前 wiki artifact | 同步内容 | SHA-256 |
|---|---|---|
| [Research proposal](../../kvcache-wiki/raw/research/CXL_KV_Research_Proposal_2026-09-28.md) | 当前中文 PROPOSAL；五面板职责说明；调整 intro 与相对链接 | `23ae8c137397ff36e9fd3add6c584589e1885e103ca86ce291c71adae41e6bb2` |
| [Architecture HTML](../../kvcache-wiki/raw/research/CXL_KV_Architecture_v2.html) | 英文图、中文模块责任/源码映射、架构契约、完整 INDEX_LIFECYCLE、实验入口；无外部页面依赖 | `b034840bc6966cc4659e3a8d5ac73ac27d140033b8db84dc38a7ad50a28c2259` |
| [Architecture PNG](../../kvcache-wiki/raw/research/figures/architecture.png) | 与本仓库 `docs/figures/architecture.png` 字节一致 | `551717a12acaaa2cedf1f5a0724dd377483adffe2560cca7953c179d21d0d688` |

同步更新 wiki 的 index、overview、sparse-attention-cxl-anns 专题与 append-only log：主问题调整为持续多轮 IVF 的追加、复用/替换和有界内存；固定选集的执行分流保留为次级验证轴，历史 graph/learned-selector 路径明确标记为设计对照。

实际检查仅覆盖文档：HTML 标签闭合、20 个唯一 ID（10 个架构章节、7 个生命周期章节及 3 个入口）、263 处本地引用的目标文件与 HTML anchor、PNG 字节一致性及 wiki `git diff --check`。没有浏览器渲染检查，也没有新增测试、论文检索、模型运行、质量或硬件实验。
