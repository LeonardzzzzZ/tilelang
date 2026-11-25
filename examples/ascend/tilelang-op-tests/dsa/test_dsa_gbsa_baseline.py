# -*- coding: utf-8 -*-
"""DSA 性能基准: tilelang DSA kernel vs ops-transformer GBSA (GenericBlockSparseAttention).

对标算子 (cann/ops-transformer, master):
    - 主算子:    attention/generic_block_sparse_attention
      https://gitcode.com/cann/ops-transformer/tree/master/attention/generic_block_sparse_attention
    - 配套参数:  attention/generic_block_sparse_attention_metadata (前置 metadata 生成)
      https://gitcode.com/cann/ops-transformer/tree/master/attention/generic_block_sparse_attention_metadata

依赖
    - cann_ops_transformer 包 (来自 ops-transformer 仓库, pip 或源码安装;
      DSA fork 验证时使用 ops-transformer commit aaad76f2b1a2f0d4aaf2a5fd531cff7aa9c1cc68);
    - 未安装时本模块的用例优雅 skip, 不影响 dsa/test_dsa.py 主套件。

对标口径 (与 DSA fork README 的 "GBSA complete-call resident" 一致)
    - GBSA 侧被计时的 callable = 布局转换 (BNSD->TND) + 索引适配 (mask 折叠)
      + metadata 生成 + 主算子调用, 输入均在 NPU 上, metadata 每次调用重新生成
      (算子契约要求与当前调用配对);
    - tilelang 侧被计时的 callable = DSA kernel 调用;
    - ratio = GBSA 时延 / TileLang 时延, > 1 表示 tilelang 更快 (fork README 记录
      其测量点均 > 1.0x);
    - 计时 backend 由环境变量 DSA_BENCH_BACKEND 控制 (默认 msprof_detail, 可换 wall
      获得端到端口径), 两侧使用同一 backend;
    - GBSA 侧计时参数 5 预热 / 20 重复, tilelang 侧 30 / 100 (各自来源口径)。

语义对齐方式
    - 布局: GBSA 侧使用 TND (实测服务器 CANN 9.2.0.B120 的 metadata 算子仅支持
      TND qInputLayout, 传 BNSD 报 EZ1001); DSA 的 Q/K/V [B,64,S,256] 经
      permute+reshape 转为 (B*S, 64, 256) / (B*T, 64, 256), 布局转换计入 GBSA 侧
      计时 (bridge 口径); 输出 TND (B*sq, 64, 256) 直接 view 回 [B,sq,64,256];
    - 稀疏粒度: block_shape=[1,1] (token 级稀疏), 对齐 DSA 的 per-position top-k
      (950 非量化 BF16 支持 block_shape=[1,Y], 1<=Y<=128);
    - mask: DSA 的逐位置 bool mask 按 fork NativeGBSA adapter 口径折叠进稀疏索引
      (选中位置 ∩ mask, 升序排序前压, 尾部填 -1, count=有效位置数);
    - 其余: mask_mode=0 (非 causal), quant_mode=0 (BF16), softmax_scale=0.0625,
      layout_sparse_pattern=4 (唯一支持值), groupSize=64/64=1。

用法
    pytest dsa/test_dsa_gbsa_baseline.py -v -s              # 全部基准点 (单轮快验)
    pytest dsa/test_dsa_gbsa_baseline.py -v -k "t4096"      # 按 shape 过滤
    DSA_GBSA_ROUNDS=3 pytest dsa/test_dsa_gbsa_baseline.py -v -s   # 多轮 (中位数+范围, 交替顺序)
    DSA_GBSA_ASSERT_FASTER=1 DSA_GBSA_ROUNDS=3 pytest ...   # 验收模式: 断言 tilelang 更快
    DSA_GBSA_BASELINE=0 pytest dsa/test_dsa_gbsa_baseline.py -v   # 强制跳过

    # BNSD 零拷贝变体 (需支持 BNSD qInputLayout 的新版 CANN):
    pytest dsa/test_dsa_gbsa_baseline_bnsd.py -v -s

环境变量
    DSA_BENCH_BACKEND      计时 backend, 默认 msprof_detail (两侧一致; wall 为端到端口径)
    DSA_GBSA_BASELINE      "0" 时跳过全部基准用例, 默认开启
    DSA_GBSA_ROUNDS        测量轮数, 默认 1; >1 时输出每轮结果、中位数与范围,
                           并按 tilelang/GBSA 交替顺序测量 (消除先后顺序偏差, fork 口径)
    DSA_GBSA_ASSERT_FASTER "1" 时断言 ratio > 1 (tilelang 必须优于 GBSA), 用于验收
"""

import os
import statistics
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest

try:
    # 与 pytest 收集的 test_dsa 为同一模块; 若 test_dsa 因依赖缺失 skip, 本模块同样 skip
    from test_dsa import (
        HEAD_DIM,
        INDEX_TOPK,
        NUM_HEADS,
        SOFTMAX_SCALE,
        build_dsa_kernel,
        make_inputs,
        validate_case,
    )
except Exception:  # pragma: no cover - test_dsa 自身 skip 时会抛 Skipped, 此处透传
    raise


# ---------------------------------------------------------------------------
# 1. 基准点表 (fork README 记录的测量 shape; 超大 sq=8192 点默认关闭)
# ---------------------------------------------------------------------------

# GBSA 侧计时参数 (fork 基准轮口径: 5 预热 / 20 重复)
GBSA_WARMUP = 5
GBSA_REPEAT = 20
# tilelang 侧计时参数 (fork example 口径: 30 预热 / 100 重复)
TL_WARMUP = 30
TL_REPEAT = 100

GBSA_BASELINE_CASES = [
    dict(batch_size=1, seq_len=16, kv_len=2048),   # fork README 基准点
    dict(batch_size=2, seq_len=32, kv_len=512),    # batch 维
    dict(batch_size=1, seq_len=17, kv_len=2049),   # 非对齐 T + TopK 上限
    dict(batch_size=1, seq_len=448, kv_len=4096),  # fork 主基准 shape
    # dict(batch_size=1, seq_len=1, kv_len=4097),     # 单 query (按需打开)
    # dict(batch_size=1, seq_len=8192, kv_len=8192),  # 超大 shape (按需打开, 非常慢)
]


def _baseline_id(cfg: dict) -> str:
    return f"gbsa_b{cfg['batch_size']}_sq{cfg['seq_len']}_t{cfg['kv_len']}".lower()


BASELINE_IDS = [_baseline_id(c) for c in GBSA_BASELINE_CASES]


# ---------------------------------------------------------------------------
# 2. DSA 输入 -> GBSA 输入的适配 (fork NativeGBSA adapter 的 pack_indices 口径)
# ---------------------------------------------------------------------------

def _fold_mask_into_indices(indices, mask, kv_len, num_heads=NUM_HEADS):
    """把 DSA 的 (indices, mask) 折叠成 GBSA pattern-4 的 (sparse_block_idx, count).

    DSA: indices [B, sq, Ksel] int32 (每条 query 的唯一选中位置, 全 head 共享),
         mask [B, 1, sq, T] bool (非零 = 参与注意力)。
    GBSA: sparse_block_idx [KV_N, B*sq, Ksel] int32, sparse_block_count
          [KV_N, B*sq] int32; 契约要求有效索引非负、严格升序、不重复,
          位于每行前 count 个位置, 其余填 -1 (容量内不参与计算)。

    折叠逻辑: 选中位置 ∩ mask -> 升序排序 -> 无效位填越界值再排序前压 -> 尾部置 -1。
    """
    import torch

    ksel = indices.shape[-1]
    mask_u8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    keep = mask_u8.squeeze(1).bool()                       # [B, sq, T]

    sorted_idx = torch.sort(indices.long(), dim=-1).values  # [B, sq, Ksel] 升序唯一
    keep_sorted = torch.gather(keep, 2, sorted_idx)         # [B, sq, Ksel]
    # 无效位置填越界值 (kv_len >= 所有合法索引), 排序后有效升序在前
    packed = torch.where(keep_sorted, sorted_idx, kv_len)
    packed = torch.sort(packed, dim=-1).values
    count = keep_sorted.sum(dim=-1).to(torch.int32)         # [B, sq]

    # 前 count 个位置保留有效索引, 其余置 -1
    positions = torch.arange(ksel, device=indices.device).view(1, 1, ksel)
    packed = torch.where(
        positions < count.unsqueeze(-1).long(), packed, torch.full_like(packed, -1)
    )
    packed = packed.to(torch.int32).reshape(-1, ksel)       # [B*sq, Ksel]

    # pattern-4: [KV_N, totalQBlocks, Ksel]; groupSize=1, 全 head 共享同一稀疏模式
    sparse_idx = packed.unsqueeze(0).expand(num_heads, -1, -1).contiguous()
    sparse_cnt = count.reshape(-1).unsqueeze(0).expand(num_heads, -1).contiguous()
    return sparse_idx, sparse_cnt


def _make_gbsa_call(ops, q, k, v, indices, mask, seq_len, kv_len):
    """构造 GBSA 完整调用 (布局转换 + 适配 + metadata + 主算子), 作为被计时的 callable.

    服务器 CANN 9.2.0 的 metadata 算子仅支持 TND qInputLayout, 故 Q/K/V 由 DSA 的
    [B, H, S, D] 转为 TND (B*S, H, D) / (B*T, H, D), 配 cu_seqlens 前缀和;
    转换计入被计时的调用链 (bridge 口径)。
    """
    import torch

    batch_size = q.shape[0]
    # TND Q/KV 必传存储长度前缀和 (int64, 首元素为 0)
    cu_seqlens_q = torch.arange(batch_size + 1, device=q.device) * seq_len
    cu_seqlens_kv = torch.arange(batch_size + 1, device=q.device) * kv_len
    block_shape = [1, 1]  # token 级稀疏, 对齐 DSA per-position top-k

    def call():
        # 布局适配: [B, H, S, D] -> TND (B*S, H, D); permute 后 reshape 触发拷贝
        q_tnd = q.permute(0, 2, 1, 3).reshape(-1, NUM_HEADS, HEAD_DIM)
        k_tnd = k.permute(0, 2, 1, 3).reshape(-1, NUM_HEADS, HEAD_DIM)
        v_tnd = v.permute(0, 2, 1, 3).reshape(-1, NUM_HEADS, HEAD_DIM)
        sparse_idx, sparse_cnt = _fold_mask_into_indices(indices, mask, kv_len)
        metadata = ops.generic_block_sparse_attention_metadata(
            sparse_idx,
            sparse_cnt,
            NUM_HEADS,
            NUM_HEADS,
            HEAD_DIM,
            block_shape,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=seq_len,
            max_seqlen_kv=kv_len,
            layout_q="TND",
            layout_kv="TND",
            layout_sparse_pattern=4,
            mask_mode=0,
            quant_mode=0,
            softmax_precision=1,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        out, _ = ops.generic_block_sparse_attention(
            q_tnd,
            k_tnd,
            v_tnd,
            sparse_idx,
            sparse_cnt,
            block_shape,
            metadata=metadata,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            layout_q="TND",
            layout_kv="TND",
            layout_sparse_pattern=4,
            softmax_scale=SOFTMAX_SCALE,
            mask_mode=0,
            quant_mode=0,
            softmax_precision=1,
            return_softmax_lse=False,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        return out  # TND (B*sq, H, D)

    return call


# ---------------------------------------------------------------------------
# 3. golden 采样参考 (fork reference_attention 的按行采样版, 大 shape 只算选中行)
# ---------------------------------------------------------------------------

def _reference_attention_rows(q, k, v, indices, mask, rows):
    """FP32 golden, 只计算给定的 query 行 (fork reference_attention 的采样变体)."""
    import torch

    batch_size, heads, _, dim = q.shape
    mask_u8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    out = torch.empty((batch_size, len(rows), heads, dim), dtype=torch.float32, device=q.device)
    for batch in range(batch_size):
        k_batch = k[batch]
        v_batch = v[batch]
        for out_pos, query in enumerate(rows):
            idx = indices[batch, query].long()
            gathered_k = k_batch[:, idx, :].float()
            gathered_v = v_batch[:, idx, :].float()
            scores = torch.einsum("hd,hkd->hk", q[batch, :, query, :].float(), gathered_k) * SOFTMAX_SCALE
            keep = mask_u8[batch, 0, query][idx].ne(0)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[batch, out_pos] = torch.einsum("hk,hkd->hd", probabilities, gathered_v)
    return out


def _sample_rows(seq_len, limit=64, samples=16):
    """sq <= limit 时全量比对, 否则均匀采样 samples 行 (fork 大 shape 口径)。"""
    if seq_len <= limit:
        return list(range(seq_len))
    import torch

    return torch.linspace(0, seq_len - 1, samples).long().unique().tolist()


def _bench_ms(fn, backend, warmup, repeat):
    """单次 do_bench 计时, 返回毫秒 (兼容 msprof 统计对象与 float 两种返回)."""
    from tilelang.profiler import do_bench

    prof = do_bench(fn, backend=backend, _n_warmup=warmup, _n_repeat=repeat)
    return (prof.dur_ns / 1e6) if hasattr(prof, "dur_ns") else float(prof)


def _perf_compare_and_report(tl_fn, gbsa_call, backend, batch_size, seq_len, kv_len, ksel, label=""):
    """多轮 A/B 交替测量 tilelang 与 GBSA 并输出对比结论; 返回 (tl_ms, gbsa_ms, ratio)。

    统一性能口径: tilelang 耗时以 "X.XXX ms" (3 位小数) 输出。
    环境变量: DSA_GBSA_ROUNDS (轮数, 默认 1; >1 时交替测量顺序并输出中位数+范围),
    DSA_GBSA_ASSERT_FASTER (=1 时断言 ratio > 1, tilelang 必须更快)。
    供 TND / BNSD 两个基准脚本共用。
    """
    rounds = max(1, int(os.environ.get("DSA_GBSA_ROUNDS", 1)))
    tl_times, gbsa_times = [], []
    for r in range(rounds):
        # 交替先后顺序, 消除测量顺序带来的系统性偏差 (fork 基准口径: A/B 交替)
        if r % 2 == 0:
            tl_ms_r = _bench_ms(tl_fn, backend, TL_WARMUP, TL_REPEAT)
            gbsa_ms_r = _bench_ms(gbsa_call, backend, GBSA_WARMUP, GBSA_REPEAT)
        else:
            gbsa_ms_r = _bench_ms(gbsa_call, backend, GBSA_WARMUP, GBSA_REPEAT)
            tl_ms_r = _bench_ms(tl_fn, backend, TL_WARMUP, TL_REPEAT)
        tl_times.append(tl_ms_r)
        gbsa_times.append(gbsa_ms_r)
        if rounds > 1:
            print(
                f"[perf] round {r + 1}/{rounds}: tilelang {tl_ms_r:.3f} ms | "
                f"GBSA {gbsa_ms_r:.3f} ms | ratio {gbsa_ms_r / tl_ms_r:.2f}x"
            )

    tl_ms = statistics.median(tl_times)
    gbsa_ms = statistics.median(gbsa_times)
    flops = 4.0 * batch_size * seq_len * NUM_HEADS * HEAD_DIM * ksel
    ratio = gbsa_ms / tl_ms if tl_ms > 0 else float("inf")
    tl_rng = f" (range {min(tl_times):.3f}~{max(tl_times):.3f})" if rounds > 1 else ""
    gbsa_rng = f" (range {min(gbsa_times):.3f}~{max(gbsa_times):.3f})" if rounds > 1 else ""
    print(
        f"[perf][tilelang] {label}: {tl_ms:.3f} ms | "
        f"{flops / (tl_ms * 1e-3) / 1e12:.1f} TFLOPS | 中位数{tl_rng} ({TL_WARMUP}w/{TL_REPEAT}r)"
    )
    print(f"[perf][gbsa] {label}: {gbsa_ms:.3f} ms | 中位数{gbsa_rng} ({GBSA_WARMUP}w/{GBSA_REPEAT}r, 完整调用链)")
    verdict = "tilelang faster" if ratio >= 1 else "GBSA faster"
    print(f"[perf][compare] {label}: GBSA / TileLang = {ratio:.2f}x ({verdict})")

    assert tl_ms > 0 and gbsa_ms > 0, "性能计时结果异常 (<= 0 ms)"
    if os.environ.get("DSA_GBSA_ASSERT_FASTER", "0") == "1":
        assert ratio > 1.0, (
            f"tilelang DSA 未优于 GBSA 基准: GBSA/TileLang = {ratio:.2f}x (<= 1.0), "
            f"shape: B={batch_size}, sq={seq_len}, T={kv_len}"
        )
    return tl_ms, gbsa_ms, ratio


# ---------------------------------------------------------------------------
# 4. pytest 用例
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npu_device():
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("未检测到可用 NPU (需要 torch_npu + CANN 环境)")
    return torch.device("npu")


@pytest.mark.parametrize("cfg", GBSA_BASELINE_CASES, ids=BASELINE_IDS)
def test_dsa_perf_vs_gbsa(cfg, npu_device):
    """DSA 性能基准: tilelang kernel vs GBSA 完整调用链 (适配 + metadata + 主算子)。

    流程: 编译 tilelang 侧 -> 生成输入 -> 双侧各跑一次并对 FP32 golden 交叉验证
    (大 shape 采样 16 行) -> 双侧 do_bench -> 输出 ratio (GBSA / TileLang)。
    ratio > 1 表示 tilelang 更快 (fork README 记录其测量点均 > 1.0x)。
    """
    import torch

    if os.environ.get("DSA_GBSA_BASELINE", "1") == "0":
        pytest.skip("DSA_GBSA_BASELINE=0, 跳过 GBSA 性能基准")
    try:
        import cann_ops_transformer
    except ImportError as exc:
        pytest.skip(
            f"未安装 cann_ops_transformer (ops-transformer 仓库), 跳过 GBSA 基准: {exc}"
        )
    ops = getattr(cann_ops_transformer, "ops", cann_ops_transformer)

    validate_case(cfg)
    batch_size, seq_len, kv_len = cfg["batch_size"], cfg["seq_len"], cfg["kv_len"]
    backend = os.environ.get("DSA_BENCH_BACKEND", "msprof_detail")
    print(
        f"\n[baseline] {_baseline_id(cfg)}: B={batch_size}, sq={seq_len}, T={kv_len}, "
        f"Ksel={min(INDEX_TOPK, kv_len)}, backend={backend} "
        f"(GBSA 侧=布局转换+适配+metadata+主算子, tilelang 侧=kernel)"
    )

    # tilelang 侧 kernel (编译缓存与 dsa/test_dsa.py 共享)
    tl_kernel = build_dsa_kernel(cfg)
    q, k, v, indices, mask = make_inputs(cfg, npu_device)
    mask_u8 = mask.view(torch.uint8)

    # golden (采样) 与双侧正确性交叉验证
    rows = _sample_rows(seq_len)
    expected = _reference_attention_rows(q, k, v, indices, mask, rows)

    tl_out = tl_kernel(q, k, v, indices, mask_u8)
    torch.npu.synchronize()
    tl_f = tl_out.float()[:, rows]
    max_abs_tl = (tl_f - expected).abs().max().item()
    assert torch.allclose(tl_f, expected, rtol=1e-2, atol=1e-2), (
        f"tilelang DSA 输出与 golden 不符: max_abs={max_abs_tl:.4e}"
    )

    gbsa_call = _make_gbsa_call(ops, q, k, v, indices, mask, seq_len, kv_len)
    gbsa_out = gbsa_call()
    torch.npu.synchronize()
    # GBSA 输出 TND (B*sq, H, D) -> view 回 [B, sq, H, D] 对齐 golden
    gbsa_f = gbsa_out.float().view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)[:, rows]
    max_abs_gbsa = (gbsa_f - expected).abs().max().item()
    assert torch.allclose(gbsa_f, expected, rtol=1e-2, atol=1e-2), (
        f"GBSA 输出与 golden 不符 (适配或调用参数可能有误): max_abs={max_abs_gbsa:.4e}"
    )
    print(
        f"[correctness] tilelang max_abs={max_abs_tl:.4e} | "
        f"GBSA max_abs={max_abs_gbsa:.4e} (采样 {len(rows)}/{seq_len} 行, atol=rtol=1e-2) PASS"
    )

    # 性能对比 (多轮 A/B 交替, 中位数, 详见 _perf_compare_and_report)
    tl_fn = lambda: tl_kernel(q, k, v, indices, mask_u8)  # noqa: E731
    _perf_compare_and_report(
        tl_fn, gbsa_call, backend, batch_size, seq_len, kv_len, min(INDEX_TOPK, kv_len),
        label=_baseline_id(cfg),
    )
