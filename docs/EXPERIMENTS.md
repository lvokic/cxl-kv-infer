# 首轮实验方案

状态：实验设计；本次交付没有执行这些实验，也没有生成性能结果。

## 1. 分开回答两个问题

**执行实验：** 固定 Q/K/V、逐 head 最终选集 S_h、精度、位置语义、资源预算，改变读取、传输、计算路径。

**检索实验：** 在相同模型质量约束下改变索引和候选 R；记录最终选集变化、截断误差、索引误差以及成本。相同 token budget 不自动意味着相同质量。

离线 trace 可以固定输入，不能把离线精排、缓存命中或 oracle 决策当成免费在线工作。

## 2. 执行路径矩阵

| 路径 | 目的 | 必须包含的成本 |
|---|---|---|
| selected KV 全在 HBM | 数值参照和驻留场景诊断 | attention；单独说明不含冷搬运 |
| GPU hot + CPU cold partial | CPU 简单基线 | Q D2H、冷读、CPU partial、state H2D、merge |
| GPU hot + GPU packed cold | GPU 简单基线 | gather、packet、H2D、attention、merge |
| GPU hot + logits/V packet | 精排复用分支 | 真实精排、分数打包、V 冷读、H2D、PV kernel、merge |
| whole-group 静态选择 | 评估简单路由能力 | 查表和全部执行开销 |
| group 内有限 mixed | 待验证主机制 | 规划、额外分区、两侧争用、join、merge |
| 离线 best-of-paths | 描述已有路径的选择空间 | 明确标记 oracle；不能称为在线结果 |

HBM-only 结果没有承受冷数据路径的容量约束，不能作为可直接达到的端到端速度。独立测量后取 min 也不能重建共享 CXL 链路上的真实并发成本。

## 3. 首批维度

先用一个真实模型的少量代表层；一组稳定后扩成整层，避免一次扫完整笛卡尔积。

| 变量 | 初始取值或做法 |
|---|---|
| Context | 32K / 64K / 128K，在模型支持范围内 |
| 历史 k | 256 / 1024，质量待独立校准 |
| 候选 R/k | 1 / 2 / 4，检索或复用实验使用 |
| Backing | node 1 DRAM / node 3 CXL；node 0 DRAM 作远端对照 |
| CPU workers | 从 node 1 少量物理核开始，例如 1 / 4 / 8 / 16 |
| HBM residency | 相同预算下的固定比例和真实命中分布 |
| GQA sharing | 先记录真实 union / membership，合成数据仅补角落 |
| Backing page | 初始 16 tokens；与 kernel 的 page_size=1 视图分开 |
| 布局 | 先固定 token-major，后续再独立引入语义布局 |

按 8 KV heads、d=128、BF16 计算，128K tokens 单层 KV 约 512 MiB，单个 group 约 64 MiB。后者可能被 CPU cache 大量覆盖；报告 warm-cache 情况，另用轮换工作集测真实 backing 访问，不把 cache 命中解释为 CXL 带宽。

真实 trace 需要覆盖不同 decode steps 和任务；teacher-forced replay 只用于隔离机制。质量评估还需要自由生成，不能只看离线向量误差。

## 4. 时间边界

分阶段时间：

```text
Q-ready
  → Q D2H / selection
  → candidate K lookup / gather / original-precision refine
  → final selection and K/logit handoff
  → plan / V gather / packet / H2D
  → attention or softmax-PV
  → partial merge / output-ready
```

- 单层主指标是从 Q-ready 到 output-ready 的 wall time，包含 CPU/GPU 同步和队列等待。
- CUDA event 衡量设备区间；CPU 单调时钟衡量主机区间。两者不能直接相减当作统一时间轴。
- setup、JIT、分配、CUDA graph capture 单列；在线必要的 plan/descriptor 更新仍须计入。
- 重复运行时消费结果、保持合理的并发深度；不要每个小算子都全设备同步破坏本来可有的重叠。
- 同时保留串行诊断和真实并发版本，记录 contention；两条路径不能各按独占 CXL 带宽计算。
- 第一阶段 reference/GPU adapter CLI 没有实现这些计时要求，不产出 benchmark 时间。

## 5. 字节、容量和质量

逐项记录：

- `sum_h |C_h|`、候选 union `|U_R|`、`sum_h |S_h|`、最终 union `|U_k|`。
- 逻辑 K/V bytes、实际 H2D payload、descriptor bytes；可用时记录硬件 CXL/IMC/PCIe counters。
- 无 counters 时使用 `logical_payload_bytes` / `estimated_transaction_bytes` 标签，禁止命名为 measured CXL bytes。
- 模型权重、HBM hot KV、GPU workspace、packet ring、DRAM index/codes、候选 lease、pinned buffers、canonical CXL KV 的峰值。
- 索引复制、FP32 Faiss 原向量、构建临时空间、旧/新版本重叠都计入。
- 固定选集数值误差与相对 full attention 的模型质量分别报告。

忽略命中和事务放大时：

```text
CXL cold payload = dK*bK*|U_R,coldK| + dV*bV*|U_k,coldV|
```

如果 KLease 失效造成重读，额外加入真实重读量。无跨 head overlap 且 K/V 等宽时，相对只读取最终 K+V 的放大为 `(R/k+1)/2`；R=4k 得到 2.5 倍，属于分析示例。

`logits + V` 的分析示例：d=128、BF16、每 token 被四个 heads 选择，K+V 为 512 bytes，V+FP32 logits 为 272 bytes。它没有减少此前精排 K 的读取；需测打包和额外 kernel 是否抵消 H2D payload 的节省。

## 6. 后续结果记录格式

benchmark 实现时，每个 run 目录应包含以下信息。当前 CLI 仅保存数值结果和基本摘要，尚未实现完整记录器。

```text
runs/<run-id>/
  manifest.json       # commit、环境、模型、trace hash、拓扑、placement 证据、资源预算
  config.json         # 路径、k/R、索引、CPU affinity、buffer limits
  samples.jsonl       # 单次样本、阶段区间、实际执行路径、异常/重试
  metrics.json        # 统计量和每个指标定义；非实测字段显式标记
  timeline.*          # 可选 CPU/GPU timeline
```

`runs/` 和 `data/` 默认不进 Git。提交研究结论时保存经过筛选的配置、图和分析，并能追溯到原始 run。

## 7. 继续与停止条件

先检查 whole-group / 静态最佳路径与可实现 mixed 的差距；若 planner、join、mask 处理吃掉收益，就保留简单策略。如果所有路径都被候选 K 冷读限制，优先研究检索或减少精排候选；如果主要时间在 MLP/权重，评估端到端 Amdahl 空间。

CXL 实验还要体现 DRAM 容量约束。先做明确的 DRAM budget 对照，后续再扩实际工作集；仅把能完全放进 DRAM 的数据搬到 CXL 并比较延迟，不能充分证明系统价值。
