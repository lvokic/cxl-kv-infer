# Native execution: next implementation boundary

本目录尚无 allocator 或优化 kernel。下列是 T02 / T03 的接口要求，不能当成已完成代码。

建议采用独立 C++ library + PyBind11 binding，首先只链接 libnuma / CPU 标准依赖。
CUDA host registration 可以在之后单独的 staging 模块完成。

```text
NumaBuffer.allocate(bytes, node, alignment)
    → owned buffer, requested_node, allocation_policy
NumaBuffer.residency(sample_pages)
    → actual page-node counts, observation timestamp
gather_k(candidate_ids, directory, destination_lease)
    → ready lease, logical bytes, completion
gather_v(selected_ids, directory, destination_packet)
    → ready packet, logical bytes, completion
cpu_partial(exact_logits, V_spans, head_membership)
    → normalized output + natural-log LSE
```

- `numa_alloc_onnode` 的返回指针本身不证明 mbind 成功；处理策略错误并检查实际页位置。
- 可采用匿名 mmap + mbind before first-touch + placement query。节点 ID 非连续，禁止假设是 0..N-1。
- 区分 allocation bytes、逻辑 gather bytes、cache-line 请求估计和硬件流量。
- K/V 分开；index/staging/backing 使用不同 allocation。首次触页后再做 residency 检查。
- staging 按 GPU 邻近 node 分配，使用有界 ring；CUDA registration 失败必须报错。
- PyBind 返回 view 时，Python owner 必须持有 native allocation；异步消费者结束之前不能释放。
- CPU kernel 首先实现 FP32 累积的 stable softmax + PV；精排 logits 有效时复用它们。
- CXL 读取和 kernel 实现完成前，不提供命名为 `cxl` 的伪后端。
