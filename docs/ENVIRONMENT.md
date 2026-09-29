# 环境准备与代码入口

## 1. 当前状态

2026-09-29 的当前 shell Python 未安装 NumPy、PyTorch、FlashInfer。尚未建立并验证 GPU 软件组合，也未编译 native 模块。本仓库的 optional dependencies 只声明名称；没有把任意最新版当成已验证配置。

用户提供的 nvidia-smi 显示 A100 PCIe 40GB、driver 595.71.05 和已有约 15.6 GiB 显存占用。`CUDA Version: 13.2` 表示 driver 能力；不能据此判断已安装的 CUDA toolkit 或某个 PyTorch wheel 的兼容性。不要操作其他项目的进程或虚拟环境。

## 2. 隔离环境的准备顺序

先使用本仓库 `.venv` 安装 CPU core，命令见 README。之后在明确需要 GPU 实现/验证时：

1. 查看 Python 版本、`nvcc --version`、driver、现有 CUDA 库，选择官方支持的 PyTorch/FlashInfer 组合。
2. 对照下表固定的已审查源码，确认实际安装版本的 `plan/run`、dtype、backend、LSE 和 page-size 契约。
3. 在这个独立环境安装 GPU 依赖；`.[gpu]` 是依赖分组，不是经过验证的 lock。
4. 记录 `pip freeze --all`、软件 commit/wheel 版本、编译选项、GPU arch。首次成功后再建立环境锁定文件。
5. 数值验证和硬件实验按 owner 的验证请求执行，保存结果后更新交付状态。

准备好 GPU 环境后的 adapter 调用方式：

```bash
cd ~/cxl-kv-replay
.venv/bin/cxl-kv-replay replay data/synthetic/group.npz --backend flashinfer --dtype bfloat16 --output runs/flashinfer.npz
```

这会分配 GPU workspace 并执行一次 attention，但没有性能计时或 CXL placement。源码目前未运行。较大真实 trace 需要先检查 selected payload 和可用显存，当前 adapter 并非有界生产 allocator。

只读平台快照的使用方法：

```bash
python3 scripts/inspect_platform.py --output runs/platform.json
```

脚本使用标准库，保存命令返回码；缺失命令记录为 unavailable。它不会测带宽、迁移页面或启动模型。

## 3. NUMA 放置计划

见 [platform.json](../configs/platform.json)。此前只读检查显示 GPU BDF `0000:8b:00.0` 的 NUMA node 为 1，当前 PCIe 为 16 GT/s ×16。node 3 为 owner 确认的 memory-only CXL。不要从相同 NUMA distance 推断相同延迟/带宽或具体 root complex。

| 对象 | 起始策略 |
|---|---|
| Canonical cold K/V | 单独分配在 node 3 |
| Index / codes / directory | node 1 DRAM |
| Pinned staging ring | node 1 DRAM，有界容量 |
| CPU workers | 先选 node 1 的物理核，例如 32–39，再扩展 |
| 对照 backing | node 1 DRAM、node 0 DRAM；其余策略保持一致 |

不使用整个进程 `--membind=3` 来代替分对象控制。mbind/numa allocation 必须检查策略失败；首次触页后查询实际 residency。记录 automatic NUMA balancing / page migration 影响；必要时重复检查，不能只记申请的 node。

GPU mapped CXL / host registration 不是初版前提。普通 Type-3 memory 没有 near-memory compute；本项目先采用 CPU 访问 CXL，然后 CPU partial 或 staging→GPU 的路径。

## 4. 已审查的上游版本与入口

以下是上轮源码阅读时记录的 commit，**不是已安装或编译通过的版本**。链接指向固定源码，升级后重新审查。

| 项目 | Commit | 入口 / 用途 |
|---|---|---|
| FlashInfer | `795a96b229185fb47a84884880ad7aa171747f30` | [decode.py](https://github.com/flashinfer-ai/flashinfer/blob/795a96b229185fb47a84884880ad7aa171747f30/flashinfer/decode.py)：paged wrapper；[cascade.cuh](https://github.com/flashinfer-ai/flashinfer/blob/795a96b229185fb47a84884880ad7aa171747f30/include/flashinfer/attention/cascade.cuh)：base-2 LSE merge |
| Faiss | `6c8378e4f1b8d7a3a7986c2ca216923221fe6954` | [IndexScalarQuantizer.h](https://github.com/facebookresearch/faiss/blob/6c8378e4f1b8d7a3a7986c2ca216923221fe6954/faiss/IndexScalarQuantizer.h)：IVF-SQ8；[IndexIVF.h](https://github.com/facebookresearch/faiss/blob/6c8378e4f1b8d7a3a7986c2ca216923221fe6954/faiss/IndexIVF.h)：inverted-list 扩展 |
| RetroInfer / RetrievalAttention | `03f912c6e917c380d9d90c5ec85bb0f161ba53ef` | [model_hub/LLM.py](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/model_hub/LLM.py)：layer_decode；[retroinfer_cache.py](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/cache_hub/retroinfer_cache.py)：buffer 与 attention |
| RoarGraph | `7c9ef95dcb9b1022a2e3b8dfee55409ad73c605e` | [src/index_bipartite.cpp](https://github.com/matchyc/RoarGraph/blob/7c9ef95dcb9b1022a2e3b8dfee55409ad73c605e/src/index_bipartite.cpp)：投影图构建与搜索 |
| SGLang | `069523f999cb5a27d4c2da4ed690a42f6042fd67` | [hisparse_hook.py](https://github.com/sgl-project/sglang/blob/069523f999cb5a27d4c2da4ed690a42f6042fd67/python/sglang/srt/arg_groups/hisparse_hook.py)：模型约束；后期 serving 参考 |

## 5. 容易误用的接口

- FlashInfer 的一些 GQA sparse backend 要求组内共享 mask。本项目先用独立 head 逻辑行，不把 union 当成所有 heads 的逻辑选择。
- A100 是 SM80；当前 adapter 限定常规 `fa2`/CUDA-core decode，不依赖 Hopper/Blackwell 特性。
- `merge_state` 接收 normalized partial output 和 LSE，不接收未经归一化的 accumulator。该路径的 LSE 为 base 2。
- Faiss 的 IVF constructor 默认 metric 可能是 L2，必须明确 inner product；IVF-Flat 内部存 FP32 副本，默认分配也不保证 NUMA 位置。
- `microsoft/RetrievalAttention` 所查主线为 RetroInfer，包含 estimation zone；它的算法与逐 head fixed-top-k 实验不同。
- 所查 SGLang HiSparse 限定 DSA / DeepSeek V4 / MiniMax M3 等模型，不能直接给普通 Llama GQA 打开开关。
- RoarGraph 的 offline construction / insertion 不等于完整的 serving append 协议。初版使用 immutable base + exact delta。
