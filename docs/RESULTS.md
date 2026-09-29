# 已执行结果

更新：2026-09-29。只记录实际运行的范围。代码存在、数值验证、模型质量和硬件性能属于不同证据。

## 合成 attention

FP64 是 CPU 数学参考，用于减少参考自身的舍入误差；它不代表生产 CPU executor。FlashInfer 是独立 attention 项目，消费 PyTorch tensor。

| 对照 | 固定逐元素容差 | 实际执行 |
|---|---|---|
| FP64 vs scalar oracle / partial merge | output 与 natural-log LSE: `atol=rtol=1e-12` | 33 个合成 trace、空分区和 merge 检查 |
| BF16 FlashInfer | output `2e-3 / 8e-3`；LSE `2e-4 / 2e-6` | CPU/GPU 共 644 项检查通过 |
| FP16 FlashInfer | output `2e-4 / 1e-3`；LSE `2e-4 / 2e-6` | CPU/GPU 共 644 项检查通过 |

最大 output 绝对误差：FP64 `5.24e-14`、BF16 `1.9493e-3`、FP16 `2.4414e-4`；最大 natural-log LSE 误差分别为 `5.6843e-14`、`1.9041e-5`、`1.9041e-5`。合成输入的输出范围限定在 V∈[-1,1]。

原始记录：[CPU run](../runs/numerics-cpu-20260929-01/summary.json) · [GPU run](../runs/numerics-gpu-20260929-02/summary.json)。首次 GPU JIT link 失败记录在 [失败目录](../runs/numerics-gpu-20260929-01/)。

## 真实模型集成

使用固定 revision 的 SmolLM2-135M-Instruct，30 layers、9 Q heads / 3 KV heads、d=64。预先固定的整模型 logits 阈值为逐元素 `atol=0.1, rtol=0.01` 且 relative L2≤0.005。手写 cached greedy 与 Transformers 原生 `generate` 的 token 和 logits 完全一致。

| 运行 | 检查结果 |
|---|---|
| [FP16](../runs/inference-validation-fp16-20260929-02/summary.json) | 20/20；full/sparse SDPA 和 FlashInfer、FP64 capture/replay、cache 对照、head mask |
| [FP32](../runs/inference-validation-fp32-20260929-01/summary.json) | 18/18；SDPA / FP64 控制 |
| [BF16](../runs/inference-validation-bf16-20260929-02/summary.json) | 16/20；四个严格 logits 比较失败，记录未覆盖 |
| [单层 FlashInfer](../runs/inference-smol-layer-flashinfer-20260929-01/summary.json) | 2 prompts × 8 tokens；layer 0 的 3 个 KV groups 都路由到 FlashInfer，导出 6 traces |
| [本地权重与 CPU](../runs/inference-local-cpu-20260929-01/summary.json) | FP32 CPU SDPA、raw prompt 和本地 safetensors 加载运行成功 |
| [入口与 capture 检查](../runs/inference-final-checks-20260929-01/summary.json) | 14 项：无重依赖 import、非法配置、整层实际路由、六份 trace 的 FP64 参考和 head mapping |

FP16 full FlashInfer / SDPA logits 最大绝对差 `0.044922`、relative L2 `0.001206`；同一 sparse 选集比较为 `0.070313 / 0.002407`。真实单 attention 对 FP64 的最大绝对误差为 full `1.473e-5`、sparse `2.601e-5`。六份整层 FP16 capture 对 FP64 的最大绝对差 `1.159e-4`。FP32 cache 对照最大绝对差 `2.241e-5`，full 与 sparse attention 对照均为 `1.621e-5`。

BF16 四项失败的 relative L2 为 `0.011746`、`0.014606`、`0.014606`、`0.010097`；未根据结果调宽阈值。原生 generate 一致，单 attention 数值项通过，full FlashInfer top-1 与 SDPA 一致。这些对照尚不足以作严格因果归因。

## 性能和质量边界

上述运行用于集成与数值检查。同步单请求耗时、JIT 时间、PyTorch allocator 峰值是诊断数据。没有收集任务质量分数、稳定延迟/吞吐 benchmark、CXL 页驻留证据、硬件流量或 CXL 性能结果。实验文件保存在 `runs/`，不入 Git；下一步见 [RESEARCH.md](RESEARCH.md)。
