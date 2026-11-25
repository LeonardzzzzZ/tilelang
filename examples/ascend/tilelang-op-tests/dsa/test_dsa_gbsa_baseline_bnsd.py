# -*- coding: utf-8 -*-
"""DSA 性能基准 (BNSD 零拷贝变体): 供支持 BNSD qInputLayout 的新版 CANN 使用.

背景
    CANN 9.2.0.B120 的 GBSA metadata 算子仅支持 TND qInputLayout (BNSD 报
    EZ1001), 因此主基准脚本 test_dsa_gbsa_baseline.py 使用 TND 布局,
    Q/K/V 需经 permute+reshape 转换 (计入 GBSA 侧计时)。
    本脚本保留 "原方案": DSA 的 Q/K/V [B, 64, S, 256] 即 GBSA 的 BNSD 布局,
    原张量零拷贝直传, 无布局转换开销 —— 适用于支持 BNSD 的新版 CANN。
    在仅支持 TND 的 CANN 上运行时, 对应用例优雅 skip (识别 EZ1001 /
    qInputLayout 报错), 不影响测试套件其余部分。

与主基准的差异 (其余口径完全一致, 共用其基准点表与计时逻辑)
    - layout_q = layout_kv = "BNSD"; BNSD 必传 seqused_q / seqused_kv
      (int32 [B]) 与 max_seqlen_q, 不使用 cu_seqlens;
    - GBSA 侧被计时的 callable = 索引适配 (mask 折叠) + metadata 生成 +
      主算子调用 (无布局转换);
    - 输出 BNSD [B, H, sq, D] permute(0,2,1,3) 后与 golden [B, sq, H, D] 对齐。

用法 (新版 CANN 环境)
    pytest dsa/test_dsa_gbsa_baseline_bnsd.py -v -s                    # 全部基准点
    DSA_GBSA_ROUNDS=3 DSA_GBSA_ASSERT_FASTER=1 pytest dsa/test_dsa_gbsa_baseline_bnsd.py -v -s
    pytest dsa/test_dsa_gbsa_baseline.py dsa/test_dsa_gbsa_baseline_bnsd.py -v -s   # TND+BNSD 一起跑

环境变量: 与主基准脚本共用 (DSA_BENCH_BACKEND / DSA_GBSA_BASELINE /
DSA_GBSA_ROUNDS / DSA_GBSA_ASSERT_FASTER)。
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest

from test_dsa import (
    HEAD_DIM,
    INDEX_TOPK,
    NUM_HEADS,
    SOFTMAX_SCALE,
    build_dsa_kernel,
    make_inputs,
    validate_case,
)
from test_dsa_gbsa_baseline import (
    BASELINE_IDS,
    GBSA_BASELINE_CASES,
    _baseline_id,
    _fold_mask_into_indices,
    _perf_compare_and_report,
    _reference_attention_rows,
    _sample_rows,
)


# ---------------------------------------------------------------------------
# 1. BNSD 调用封装 (零拷贝原方案)
# ---------------------------------------------------------------------------

def _make_gbsa_call_bnsd(ops, q, k, v, indices, mask, seq_len, kv_len):
    """构造 GBSA BNSD 调用 (Q/K/V 原张量零拷贝直传): 适配 + metadata + 主算子."""
    import torch

    # BNSD 必传: 每 batch 实际 Q/KV 长度 (int32 [B]) 与 Q 的 shape 容量
    seqused_q = torch.full((q.shape[0],), seq_len, dtype=torch.int32, device=q.device)
    seqused_kv = torch.full((q.shape[0],), kv_len, dtype=torch.int32, device=q.device)
    block_shape = [1, 1]  # token 级稀疏, 对齐 DSA per-position top-k

    def call():
        sparse_idx, sparse_cnt = _fold_mask_into_indices(indices, mask, kv_len)
        metadata = ops.generic_block_sparse_attention_metadata(
            sparse_idx,
            sparse_cnt,
            NUM_HEADS,
            NUM_HEADS,
            HEAD_DIM,
            block_shape,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            max_seqlen_q=seq_len,
            max_seqlen_kv=kv_len,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_sparse_pattern=4,
            mask_mode=0,
            quant_mode=0,
            softmax_precision=1,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        out, _ = ops.generic_block_sparse_attention(
            q,
            k,
            v,
            sparse_idx,
            sparse_cnt,
            block_shape,
            metadata=metadata,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_sparse_pattern=4,
            softmax_scale=SOFTMAX_SCALE,
            mask_mode=0,
            quant_mode=0,
            softmax_precision=1,
            return_softmax_lse=False,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        return out  # BNSD [B, H, sq, D]

    return call


# ---------------------------------------------------------------------------
# 2. pytest 用例
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


def _gbsa_bnsd_unsupported(exc) -> bool:
    """判断异常是否为当前 CANN 不支持 BNSD qInputLayout (EZ1001)."""
    text = str(exc)
    return "qInputLayout" in text or "EZ1001" in text


@pytest.mark.parametrize("cfg", GBSA_BASELINE_CASES, ids=BASELINE_IDS)
def test_dsa_perf_vs_gbsa_bnsd(cfg, npu_device):
    """DSA 性能基准 (BNSD 零拷贝变体): 需支持 BNSD qInputLayout 的新版 CANN。

    流程与主基准一致: 编译 tilelang 侧 -> 双侧对 FP32 golden 交叉验证 ->
    多轮 A/B 交替计时 -> 输出 ratio (GBSA / TileLang)。
    仅支持 TND 的 CANN 上本用例优雅 skip。
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
        f"\n[baseline-bnsd] {_baseline_id(cfg)}: B={batch_size}, sq={seq_len}, T={kv_len}, "
        f"Ksel={min(INDEX_TOPK, kv_len)}, backend={backend} "
        f"(GBSA 侧=BNSD 零拷贝+适配+metadata+主算子, tilelang 侧=kernel)"
    )

    # tilelang 侧 kernel (编译缓存与 dsa/test_dsa.py 共享)
    tl_kernel = build_dsa_kernel(cfg)
    q, k, v, indices, mask = make_inputs(cfg, npu_device)
    mask_u8 = mask.view(torch.uint8)

    # golden (采样) 与 tilelang 侧正确性
    rows = _sample_rows(seq_len)
    expected = _reference_attention_rows(q, k, v, indices, mask, rows)
    tl_out = tl_kernel(q, k, v, indices, mask_u8)
    torch.npu.synchronize()
    tl_f = tl_out.float()[:, rows]
    max_abs_tl = (tl_f - expected).abs().max().item()
    assert torch.allclose(tl_f, expected, rtol=1e-2, atol=1e-2), (
        f"tilelang DSA 输出与 golden 不符: max_abs={max_abs_tl:.4e}"
    )

    # GBSA BNSD 侧: 不支持时优雅 skip (仅支持 TND 的旧版 CANN)
    gbsa_call = _make_gbsa_call_bnsd(ops, q, k, v, indices, mask, seq_len, kv_len)
    try:
        gbsa_out = gbsa_call()
    except RuntimeError as exc:
        if _gbsa_bnsd_unsupported(exc):
            pytest.skip(
                f"当前 CANN 的 GBSA 不支持 BNSD qInputLayout (仅 TND), "
                f"跳过 BNSD 变体 (请用主基准脚本 test_dsa_gbsa_baseline.py): {exc}"
            )
        raise
    torch.npu.synchronize()
    # GBSA 输出 BNSD [B, H, sq, D] -> [B, sq, H, D] 对齐 golden
    gbsa_f = gbsa_out.permute(0, 2, 1, 3).float()[:, rows]
    max_abs_gbsa = (gbsa_f - expected).abs().max().item()
    assert torch.allclose(gbsa_f, expected, rtol=1e-2, atol=1e-2), (
        f"GBSA 输出与 golden 不符 (适配或调用参数可能有误): max_abs={max_abs_gbsa:.4e}"
    )
    print(
        f"[correctness] tilelang max_abs={max_abs_tl:.4e} | "
        f"GBSA max_abs={max_abs_gbsa:.4e} (采样 {len(rows)}/{seq_len} 行, atol=rtol=1e-2) PASS"
    )

    # 性能对比 (多轮 A/B 交替, 中位数; 逻辑与主基准共用)
    tl_fn = lambda: tl_kernel(q, k, v, indices, mask_u8)  # noqa: E731
    _perf_compare_and_report(
        tl_fn, gbsa_call, backend, batch_size, seq_len, kv_len, min(INDEX_TOPK, kv_len),
        label=_baseline_id(cfg),
    )
