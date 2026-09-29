# 开发与运行

本仓库只使用自己的 `.venv`。环境和硬件配置是可追溯的起点；实测结果见
[RESULTS.md](RESULTS.md)，上游和研究输入版本见 [PROVENANCE.md](PROVENANCE.md)。

## 当前环境

| 部件 | 版本 / 配置 |
|---|---|
| Python / uv | 3.12.14 / 0.12.19 |
| NumPy | 2.5.3 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0，含 SM80 |
| FlashInfer | 0.7.0 (`cu13`) |
| Transformers / Hub | 4.57.6 / 0.36.2 |
| tokenizers / safetensors | 0.22.2 / 0.8.0 |
| CUDA compiler / JIT linker | nvcc 13.0.88 / nvjitlink 13.0.88 |

依赖和哈希分别保存在 `requirements/{core,gpu,inference}*.lock`；实际安装清单与原始安装
记录在 `runs/environment-20260929T054433Z/`。修改依赖时改 `.in` 并重新生成锁文件，
不要在共享环境中无记录升级。

当前 checkout 已就绪。如需重建完整推理环境：

```bash
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python --torch-backend cu130 --require-hashes \
  -r requirements/inference-cu130.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

仅 CPU trace 工具可以只安装 `requirements/core.lock`，再 editable install；
trace 包和 `cxl-kv-infer --help` 不导入 PyTorch / Transformers。CUDA 命令通过
`scripts/with-env.sh` 设置仓库 CUDA 13.0、SM80、JIT link alias 与本地编译缓存。
启动器不设置 GPU 可见性或 NUMA 放置。

## 运行模型

模型模块在 [`src/module/`](../src/module/README.md)。默认环境中已下载
HuggingFaceTB/SmolLM2-135M-Instruct，固定 revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`。权重缓存在忽略的 `data/models/`，run 输出在忽略的
`runs/`。模型/环境结果不能因此视为 CXL placement、质量评估或性能 benchmark。

```bash
bash scripts/with-env.sh cxl-kv-infer \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-files-only \
  --dtype float16 --backend flashinfer --selector exact-topk \
  --top-k 2 --sink-tokens 1 --window-tokens 1 --layers 0 --groups 0 \
  --prompt 'Explain why the sky is blue in one short sentence.' \
  --max-new-tokens 24 --capture --gpu-memory-fraction 0.1 \
  --output-dir runs/inference-new
```

已有输出目录会报错。`--model` 也接受含 config/tokenizer/safetensors 的本地目录；
`--prompts-jsonl configs/inference-prompts.jsonl` 用于顺序运行多个 prompts，每次请求有新的 cache。
默认是 BF16；目前跨后端严格整模型检查通过的是 FP16。CPU 控制运行显式指定
`--device cpu --dtype float32 --backend sdpa`。CUDA 不可用时会失败，不会悄悄改走 CPU。

`--backend sdpa --selector full` 是 dense 对照；`--groups all` 将目标 layer 的替换扩为所有
KV groups。`--layers all --groups all` 可把所有层交给该后端，但全模型 sparse quality 尚未
验证。capture 默认发生在第一个 decode forward，产物包含 post-RoPE Q/K、V、逐 head CSR、
模型 revision、输入/源码 hash 和 attention output；早停可能使目标 capture step 未执行。

选择器包括 `full`、`exact-topk`、`quest` 和 `retr-attn`。`quest` 的 `--top-k` 是 token budget，
并由 `--quest-chunk-size` 控制 chunk 大小；它按 MInference Quest 的 chunk upper bound 选块。
`retr-attn` 的 `--top-k` 是历史候选数，首尾驻留预算由 `--retrieval-sink-tokens`（默认 128）和
`--retrieval-recent-tokens`（默认 512）控制。当前实现以 CPU NumPy 精确 Flat-IP 搜索生成候选，
并在后端按原始 K 重算 attention logits；不是上游 `papyfaiss` 或其异步 cache/FlashAttention
执行路径。Quest chunk summary 当前每次 decode 重建；检索 selection 的 host K/Q copy bytes
会随运行记录保存。

## 运行 trace 工具

```bash
.venv/bin/cxl-kv-replay --help
.venv/bin/cxl-kv-replay make-synthetic --output data/synthetic/group.npz
.venv/bin/cxl-kv-replay inspect data/synthetic/group.npz
.venv/bin/cxl-kv-replay replay data/synthetic/group.npz --output runs/reference.npz
```

默认 replay 是 FP64 数学参考。synthetic 输入用于接口和数值对照；不会将 KV 分配到 CXL。

## 平台起点

硬件静态记录在 [`configs/platform.json`](../configs/platform.json)，用下面的脚本重新读取拓扑：

```bash
bash scripts/with-env.sh python scripts/inspect_platform.py --output runs/platform-new.json
```

已观察到 A100 PCIe 40 GB 位于 NUMA node 1，node 3 由 owner 标记为 memory-only CXL。
平台快照和静态配置都不能证明某个 KV allocation 实际驻留在那里。实验要对每种 allocation
单独 pin/first-touch/query residency，不要将整个进程绑定到 node 3。
