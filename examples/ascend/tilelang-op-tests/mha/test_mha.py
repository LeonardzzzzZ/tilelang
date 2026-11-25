# -*- coding: utf-8 -*-
"""MHA (Multi-Head Attention) 算子 pytest 测试脚本 (自包含, 不依赖原测试工程).

来源参考 (dangyw18/tilelang fork, commit 49a5122):
    - kernel 实现:   examples/ascend/flash_attention/core.py       (flash_attention_fwd)
    - 封装与参考实现: examples/ascend/flash_attention/example_mha.py
    - 正确性用例:    examples/ascend/flash_attention/test_mha.py
    仓库地址: https://github.com/dangyw18/tilelang/tree/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/

说明
    O = softmax(Q @ K^T / sqrt(D)) @ V, Q/K/V: [seq, D]。
    与 GQA 用的 kernel 不同, 本 fork 支持 动态 shape (T.dynamic):
    q_len 运行时可变 (允许非 128 对齐, GM 尾部 clamp + L1 零填充), 一次编译
    可在多个 shape 上运行; kv_len 仍要求 128 对齐。
    静态 (int) 编译时 q_len / kv_len 均要求 128 对齐 (kernel 内 assert)。

运行环境
    - Ascend NPU 服务器: CANN + torch_npu + tilelang (tilelang.ascend)
    - 实际测试卡为 28 个 cube 核心
    - 建议配合 TILELANG_DISABLE_CACHE=1 保证每次重新编译 (可选)

运行方式
    pytest mha/test_mha.py -v -s                 # 全部用例 (correctness + perf)
    pytest mha/test_mha.py -v -k "not perf"      # 跳过性能用例
    pytest mha/test_mha.py -v -k "dyn"           # 只跑动态 shape 用例
    pytest mha/test_mha.py -v -k "q4097"         # 按 case id / shape 过滤

kernel 约束 (不满足会在用例表中直接报错)
    1. D 固定为 128 (SIMD softmax NZ 打包路径限制);
    2. kv_len 必须能被 block_kv = 128 整除 (动态/静态均要求);
    3. 静态 (dynamic=False) 时 q_len 也必须能被 block_q = 128 整除;
       动态 (dynamic=True) 时 q_len 仅要求为正 (允许非对齐, 如 4097);
    4. num_blocks 显式指定时要求静态 q_len, 且 M tiles (ceil(q_len/128))
       必须能被 num_blocks 整除; 默认 None = 每 core 一个 M tile
       (此时 NUM_BLOCKS = M tiles, M tiles=28 的 shape 天然用满 28 核);
    5. 输入 dtype 固定 bfloat16 (fp32 累加); out_dtype 支持
       float32 (fork MHA 默认) / bfloat16 / float16。

测试规格 (通过用例默认值承载, 需变更时在单个 case 中显式指定):
    dtype = bfloat16, out_dtype = float32 (fork MHA 口径),
    精度容差 atol = rtol = 5e-2 (MHA 规格, 对齐 fork test_mha 的 rel<0.05 判据口径)。

环境变量
    MHA_BENCH_BACKEND    性能计时 backend, 默认 msprof_detail, 可换 wall 等
    MHA_BENCH_TORCH      性能用例是否附带 torch SDPA 基准, 默认 1 (开启)

shape 泛化用例表 (MHA_CASES) 当前为占位集合, 覆盖
    静态 小/中/大、28 核适配 (M tiles 为 28 的倍数)、动态 shape (含非对齐 q_len);
    具体泛化规格确认后直接替换/扩充该表即可。
"""

import math
import os
import sys
from dataclasses import dataclass

import pytest

try:
    import torch
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S
    from tilelang.layout import make_ascend_compact_nz_layout
    from tilelang.profiler import do_bench
except ImportError as exc:  # pragma: no cover - 仅在依赖缺失的环境触发
    pytest.skip(f"缺少运行依赖 (tilelang / torch), 跳过 MHA 测试: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. Kernel 实现
#    逐字复制自 dangyw18/tilelang examples/ascend/flash_attention/core.py @ 49a5122,
#    包括 FwdTiling 与 flash_attention_fwd (支持动态 q_len/kv_len, 含 return_lse 路径)。
#    如需同步上游 kernel 更新, 整段替换本节即可, 请勿单独改动实现细节。
#    注意: 与 gqa/test_gqa.py 中的 kernel 是不同版本 (本版支持动态 shape,
#    Q 每 KV 迭代重载 L0A, softmax 先缩放写回, 无 unit_flag_ctrl), 勿混用。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FwdTiling:
    block_q: int = 128
    block_kv: int = 128
    num_stages: int = 3


def flash_attention_fwd(
    q_len,
    kv_len,
    head_dim=128,
    *,
    out_dtype="bfloat16",
    softmax_scale=None,
    num_blocks=None,
    tiling=None,
    manual_schedule=False,
    return_lse=False,
):
    tiling = tiling or FwdTiling()

    # Expand config into compile-time local scalars so the hygienic T.macro
    # definitions below capture plain values, not dataclass attribute reads.
    BR = tiling.block_q
    BC = tiling.block_kv
    PIPELINE_STAGES = tiling.num_stages
    ROWS = BR // 2
    VL = 64  # Ascend fp32 SIMD lane count
    D = head_dim
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    # Implementation constraints of the SIMD softmax-packing path. These are
    # real limits of the current kernel, not tunables — validate explicitly
    # rather than advertise flexibility the code does not have.
    assert head_dim == 128, "The SIMD packing path requires head_dim == 128."
    assert BR % 2 == 0
    assert BC == 2 * VL
    if isinstance(q_len, int):
        assert q_len > 0 and q_len % BR == 0
    if isinstance(kv_len, int):
        assert kv_len > 0 and kv_len % BC == 0

    # Ascend copies clamp GM tails and zero-pad partial L1 tiles.
    NUM_M_TILES = (q_len + BR - 1) // BR
    NUM_KV_BLOCKS = kv_len // BC
    if num_blocks is None:
        NUM_BLOCKS = NUM_M_TILES
        M_TILES_PER_BLOCK = 1
    else:
        assert isinstance(q_len, int), "Explicit num_blocks requires a static q_len."
        NUM_BLOCKS = num_blocks
        assert NUM_BLOCKS > 0
        assert NUM_M_TILES % NUM_BLOCKS == 0, "M tiles must divide evenly across cores."
        M_TILES_PER_BLOCK = NUM_M_TILES // NUM_BLOCKS

    is_cast_out = out_dtype != accum_dtype
    assert out_dtype in (accum_dtype, "bfloat16", "float16"), (
        f"out_dtype must be {accum_dtype} (no cast) or a 16-bit type (bfloat16/float16); got {out_dtype!r}."
    )

    # -- shared T.macros (capture the compile-time locals above) -------------

    @T.macro
    def qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
        T.copy(K_shared[0:BC, 0:D], qk_b_l0)
        T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                "bfloat16",
            )

            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")

            for r in range(0, ROWS, 2):
                s00 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full)
                s01 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full)
                s10 = S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full)
                s11 = S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full)
                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, softmax_full)
                e10 = S.vexpdif(p10, max1, softmax_full)
                e01 = S.vexpdif(p01, max0, softmax_full)
                e11 = S.vexpdif(p11, max1, softmax_full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")

                even_bf16_0 = S.vcvt(e00, "bfloat16", softmax_full, sat=False, part=0)
                even_bf16_1 = S.vcvt(e10, "bfloat16", softmax_full, sat=False, part=0)
                odd_bf16_0 = S.vcvt(e01, "bfloat16", softmax_full, sat=False, part=1)
                odd_bf16_1 = S.vcvt(e11, "bfloat16", softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_bf16_0, "uint16x128"),
                    T.reinterpret(odd_bf16_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_bf16_1, "uint16x128"),
                    T.reinterpret(odd_bf16_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def kv_auto(
        kv,
        K,
        V,
        Q_shared,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def kv_manual(
        kv,
        K,
        V,
        Q_shared,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        # Fixed-stage solution under manual per-pipe source ordering. The
        # stage vector in source task order is:
        # [0,1,1,1,3,5,6,3,6,6,6,7,7].
        with T.Stage(0):
            T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        with T.Stage(1):
            T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
            T.copy(K_shared[0:BC, 0:D], qk_b_l0)
            T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        with T.Stage(3):
            T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])
        with T.Stage(5):
            softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        with T.Stage(6):
            pack_p(P_nz_ub, P_shared)
        with T.Stage(3):
            T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        with T.Stage(6):
            T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
            T.copy(V_shared[0:D, 0:BC], pv_b_l0)
            T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        with T.Stage(7):
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])
            accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def finalize_fp32(mt, O, O_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        T.dual_copy(O_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    @T.macro
    def finalize_cast(mt, O, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D, VL):
                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    @T.macro
    def store_lse(mt, LSE, m_ub, l_ub, lse_ub):
        with T.SimdVF(latency=256):
            lse_mask = S.pset(32, "PAT_ALL")
            lse = S.vadd(S.vln(S.vld(l_ub[0], "NORM"), lse_mask), S.vld(m_ub[0], "NORM"), lse_mask)
            S.vsts(lse_ub[0], lse, lse_mask, "NORM_B32")
        T.dual_copy(lse_ub[0:ROWS], LSE[mt * BR : (mt + 1) * BR])

    kv_step = kv_manual if manual_schedule else kv_auto

    if not return_lse:

        @T.prim_func
        def main(
            Q: T.Buffer((q_len, D), dtype),
            K: T.Buffer((kv_len, D), dtype),
            V: T.Buffer((kv_len, D), dtype),
            O: T.Buffer((q_len, D), out_dtype),
        ):
            with T.Kernel(NUM_BLOCKS) as bx:
                Q_shared = T.alloc_l1((BR, D), dtype)
                K_shared = T.alloc_l1((BC, D), dtype)
                V_shared = T.alloc_l1((D, BC), dtype)
                P_shared = T.alloc_l1((BR, BC), dtype)

                qk_a_l0 = T.alloc_l0a((BR, D), dtype)
                qk_b_l0 = T.alloc_l0b((BC, D), dtype)
                pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
                pv_b_l0 = T.alloc_l0b((D, BC), dtype)
                qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
                pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

                S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
                P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
                T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
                O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
                O_ub = T.alloc_shared((ROWS, D), accum_dtype)
                m_ub = T.alloc_shared((ROWS,), accum_dtype)
                l_ub = T.alloc_shared((ROWS,), accum_dtype)
                alpha_ub = T.alloc_shared((ROWS,), accum_dtype)
                old_m_ub = T.alloc_shared((ROWS,), accum_dtype)
                row_sum_ub = T.alloc_shared((ROWS,), accum_dtype)
                T.annotate_buffer_versions(
                    {
                        K_shared: 3,
                        V_shared: 3,
                        qk_acc_l0c: 2,
                        pv_acc_l0c: 2,
                        S_ub: 3,
                        P_nz_ub: 2,
                        alpha_ub: 3,
                        O_tmp_ub: 2,
                    }
                )

                for local_mt in T.serial(M_TILES_PER_BLOCK):
                    mt = bx * M_TILES_PER_BLOCK + local_mt
                    T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])

                    init_m_l(m_ub, l_ub)
                    clear_O_ub(O_ub)

                    for kv in T.Pipelined(
                        NUM_KV_BLOCKS,
                        num_stages=PIPELINE_STAGES,
                        annotations={"enable_offset": True},
                    ):
                        kv_step(
                            kv,
                            K,
                            V,
                            Q_shared,
                            K_shared,
                            V_shared,
                            P_shared,
                            qk_a_l0,
                            qk_b_l0,
                            pv_a_l0,
                            pv_b_l0,
                            qk_acc_l0c,
                            pv_acc_l0c,
                            S_ub,
                            P_nz_ub,
                            O_tmp_ub,
                            O_ub,
                            m_ub,
                            l_ub,
                            alpha_ub,
                            old_m_ub,
                            row_sum_ub,
                        )

                    if is_cast_out:
                        # Byte reinterpretation only: aliases O_ub, consumes no extra
                        # UB capacity or buffer versions.
                        O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                        finalize_cast(mt, O, O_ub, O_cast_ub, l_ub)
                    else:
                        finalize_fp32(mt, O, O_ub, l_ub)

        return main

    @T.prim_func
    def main_with_lse(
        Q: T.Buffer((q_len, D), dtype),
        K: T.Buffer((kv_len, D), dtype),
        V: T.Buffer((kv_len, D), dtype),
        O: T.Buffer((q_len, D), out_dtype),
        LSE: T.Buffer((q_len,), accum_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            m_ub = T.alloc_shared((ROWS,), accum_dtype)
            l_ub = T.alloc_shared((ROWS,), accum_dtype)
            alpha_ub = T.alloc_shared((ROWS,), accum_dtype)
            old_m_ub = T.alloc_shared((ROWS,), accum_dtype)
            row_sum_ub = T.alloc_shared((ROWS,), accum_dtype)
            T.annotate_buffer_versions(
                {
                    K_shared: 3,
                    V_shared: 3,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 3,
                    P_nz_ub: 2,
                    alpha_ub: 3,
                    O_tmp_ub: 2,
                }
            )

            for local_mt in T.serial(M_TILES_PER_BLOCK):
                mt = bx * M_TILES_PER_BLOCK + local_mt
                T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])

                init_m_l(m_ub, l_ub)
                clear_O_ub(O_ub)

                for kv in T.Pipelined(
                    NUM_KV_BLOCKS,
                    num_stages=PIPELINE_STAGES,
                    annotations={"enable_offset": True},
                ):
                    kv_step(
                        kv,
                        K,
                        V,
                        Q_shared,
                        K_shared,
                        V_shared,
                        P_shared,
                        qk_a_l0,
                        qk_b_l0,
                        pv_a_l0,
                        pv_b_l0,
                        qk_acc_l0c,
                        pv_acc_l0c,
                        S_ub,
                        P_nz_ub,
                        O_tmp_ub,
                        O_ub,
                        m_ub,
                        l_ub,
                        alpha_ub,
                        old_m_ub,
                        row_sum_ub,
                    )

                store_lse(mt, LSE, m_ub, l_ub, row_sum_ub)
                if is_cast_out:
                    O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                    finalize_cast(mt, O, O_ub, O_cast_ub, l_ub)
                else:
                    finalize_fp32(mt, O, O_ub, l_ub)

    return main_with_lse


# ---------------------------------------------------------------------------
# 2. MHA 封装与 torch 参考实现
#    基于 fork example_mha.py, 把 q_len / kv_len / out_dtype / num_blocks
#    显式参数化 (规格由用例表承载); fork 原始封装 mha(D, SEQ_LEN=4096)
#    等价于 mha(D_val, q_len=SEQ_LEN, kv_len=SEQ_LEN)。
# ---------------------------------------------------------------------------

def mha(D_val=128, q_len=None, kv_len=None, out_dtype="float32", num_blocks=None):
    """MHA 封装: q_len/kv_len 传 None 时使用动态符号 (一次编译多 shape 复用)."""
    if q_len is None:
        q_len = T.dynamic("q_len")
    if kv_len is None:
        kv_len = T.dynamic("kv_len")
    return flash_attention_fwd(
        q_len=q_len,
        kv_len=kv_len,
        head_dim=D_val,
        out_dtype=out_dtype,
        num_blocks=num_blocks,
    )


def ref_program(Q, K, V):
    """torch SDPA 参考实现 (MHA, 输出转 float32, 与 fork 一致)."""
    return torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
    ).squeeze(0).to(torch.float32)


# ---------------------------------------------------------------------------
# 3. shape 泛化用例表
#    TODO(shape 泛化规格): 占位集合, 规格确认后直接替换/扩充本表。
#    字段说明 (dtype / out_dtype / 容差由默认值承载测试规格, 需变更时在
#    单个 case 中显式覆盖):
#        dynamic            默认 False; True 时用 T.dynamic 编译一次,
#                           shapes 中各 shape 共用同一 kernel
#        shapes             [(q_len, kv_len), ...], 至少 1 组;
#                           动态模式 q_len 可非 128 对齐, kv_len 必须对齐;
#                           静态模式两者均须 128 对齐
#        out_dtype          输出类型, 默认 float32 (fork MHA 口径),
#                           可选 bfloat16 / float16
#        num_blocks         核数划分, 默认 None (每 core 一个 M tile;
#                           M tiles=28 的 shape 天然用满 28 核);
#                           显式指定要求静态且整除 M tiles
#        rtol / atol        精度容差, 默认 5e-2 (MHA 规格)
#        seed               随机数据种子, 默认 42
#        perf               是否纳入性能用例, 默认 False
# ---------------------------------------------------------------------------

BLOCK_Q = 128       # FwdTiling.block_q, 静态 q_len 必须被其整除
BLOCK_KV = 128      # FwdTiling.block_kv, kv_len 必须被其整除 (动态/静态均要求)
DEFAULT_RTOL = 5e-2  # 精度规格: 逐元素判据 |out-ref| <= atol + rtol*|ref|
DEFAULT_ATOL = 5e-2  # 精度规格: 绝对容差
PERF_WARMUP = 30    # 性能计时预热次数 (fork example_mha 口径)
PERF_REPEAT = 500   # 性能计时重复次数 (fork example_mha 口径)

MHA_CASES = [
    # -- 静态: 冒烟/小 shape (默认 num_blocks=None, 每 core 一个 M tile) --
    dict(shapes=[(128, 128)]),     # M tiles=1
    dict(shapes=[(256, 256)]),     # M tiles=2
    dict(shapes=[(1024, 1024)]),   # M tiles=8
    # -- 静态: fork 原始基准 shape --
    dict(shapes=[(4096, 4096)], perf=True),          # M tiles=32
    # -- 静态: 28 核适配 (M tiles 为 28 的倍数) --
    dict(shapes=[(3584, 3584)]),                      # M tiles=28, 默认调度天然 28 核
    dict(shapes=[(7168, 7168)], num_blocks=28, perf=True),  # M tiles=56, 每 block 2 tiles
    # -- 动态: fork test_mha 的 dynamic 用例 (q_len 含非对齐 4097, kv 须对齐) --
    dict(dynamic=True, shapes=[(4096, 128), (4097, 128), (4097, 256)]),
    # -- 动态: 大 shape (一次编译, 多 shape 复用) --
    dict(dynamic=True, shapes=[(4096, 4096), (7168, 7168)], perf=True),
]


# ---------------------------------------------------------------------------
# 4. 用例校验 / kernel 编译缓存
# ---------------------------------------------------------------------------

def _case_id(cfg: dict) -> str:
    """生成可读且可用于 pytest -k 过滤的 case id (小写)."""
    mode = "dyn" if cfg.get("dynamic") else "static"
    shapes = "_".join(f"q{q}-k{k}" for q, k in cfg["shapes"])
    parts = [f"mha_{mode}_{shapes}"]
    if cfg.get("out_dtype", "float32") != "float32":
        parts.append(str(cfg["out_dtype"]))
    if cfg.get("num_blocks"):
        parts.append(f"nb{cfg['num_blocks']}")
    return "_".join(parts).lower()


def validate_case(cfg: dict) -> None:
    """静态校验 shape 配置满足 kernel 硬性约束, 不满足时给出中文报错."""
    tag = _case_id(cfg)
    dynamic = bool(cfg.get("dynamic", False))
    shapes = cfg.get("shapes")

    if not shapes:
        raise ValueError(f"[{tag}] shapes 不能为空")
    for q_len, kv_len in shapes:
        if q_len <= 0 or kv_len <= 0:
            raise ValueError(f"[{tag}] q_len/kv_len 必须为正: ({q_len}, {kv_len})")
        if kv_len % BLOCK_KV != 0:
            raise ValueError(
                f"[{tag}] kv_len={kv_len} 必须能被 block_kv={BLOCK_KV} 整除 (动态/静态均要求)"
            )
        if not dynamic and q_len % BLOCK_Q != 0:
            raise ValueError(
                f"[{tag}] 静态模式 q_len={q_len} 必须能被 block_q={BLOCK_Q} 整除 "
                f"(动态模式 dynamic=True 才允许非对齐 q_len)"
            )

    nb = cfg.get("num_blocks")
    if nb is not None:
        if dynamic:
            raise ValueError(f"[{tag}] 显式 num_blocks 要求静态 q_len (kernel 约束: 动态不支持)")
        for q_len, _ in shapes:
            m_tiles = -(-q_len // BLOCK_Q)
            if m_tiles % nb != 0:
                raise ValueError(
                    f"[{tag}] num_blocks={nb} 必须整除 M tiles (ceil(q_len/{BLOCK_Q})={m_tiles})"
                )
    if cfg.get("out_dtype", "float32") not in ("float32", "bfloat16", "float16"):
        raise ValueError(
            f"[{tag}] out_dtype 仅支持 float32/bfloat16/float16, 当前 {cfg.get('out_dtype')!r}"
        )


_KERNEL_CACHE = {}


def build_mha_kernel(cfg: dict, q_len: int, kv_len: int):
    """编译 (并缓存) kernel; 动态模式整个 case 共用一个 kernel, 静态按 shape 缓存."""
    dynamic = bool(cfg.get("dynamic", False))
    out_dtype = cfg.get("out_dtype", "float32")
    nb = cfg.get("num_blocks")
    key = (
        "dyn" if dynamic else "sta",
        None if dynamic else q_len,
        None if dynamic else kv_len,
        out_dtype,
        nb,
    )
    if key not in _KERNEL_CACHE:
        validate_case(cfg)
        print(
            f"[compile] {_case_id(cfg)} 编译 shape: q_len={q_len}, kv_len={kv_len}"
            f"{' (dynamic, 一次编译多 shape 复用)' if dynamic else ''}, num_blocks={nb} ...",
        )
        func = mha(
            D_val=128,
            q_len=None if dynamic else q_len,
            kv_len=None if dynamic else kv_len,
            out_dtype=out_dtype,
            num_blocks=nb,
        )
        _KERNEL_CACHE[key] = tilelang.compile(
            func,
            out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _KERNEL_CACHE[key]


def make_inputs(q_len: int, kv_len: int, device, seed: int = 42):
    """生成 bf16 随机输入 (种子固定, 保证可复现)."""
    torch.manual_seed(seed)
    Q = torch.randn(q_len, 128, dtype=torch.bfloat16, device=device)
    K = torch.randn(kv_len, 128, dtype=torch.bfloat16, device=device)
    V = torch.randn(kv_len, 128, dtype=torch.bfloat16, device=device)
    return Q, K, V


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  # 导入即注册 npu 设备
    except Exception:
        pass
    return hasattr(torch, "npu") and torch.npu.is_available()


# 由用例表派生的 pytest 参数化数据 (需放在上面 helper 函数定义之后)
CASE_IDS = [_case_id(c) for c in MHA_CASES]
PERF_CASES = [c for c in MHA_CASES if c.get("perf")]
PERF_IDS = [_case_id(c) for c in PERF_CASES]


# ---------------------------------------------------------------------------
# 5. pytest 用例
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npu_device():
    if not _npu_available():
        pytest.skip("未检测到可用 NPU (需要 torch_npu + CANN 环境)")
    return torch.device("npu")


def test_case_configs_satisfy_kernel_constraints():
    """静态校验 MHA_CASES 用例表满足 kernel 硬性约束 (无需 NPU, 秒级)."""
    assert MHA_CASES, "MHA_CASES 用例表不能为空"
    for cfg in MHA_CASES:
        validate_case(cfg)


@pytest.mark.parametrize("cfg", MHA_CASES, ids=CASE_IDS)
def test_mha_correctness(cfg, npu_device):
    """正确性用例: tilelang kernel 输出 vs torch SDPA 参考 (bf16 输入)。

    判据 (逐元素): |out - ref| <= atol + rtol * |ref|, 默认 atol = rtol = 5e-2 (MHA 规格);
    动态 case 逐 shape 复用同一编译产物, 静态 case 按 shape 各自编译。
    """
    rtol = cfg.get("rtol", DEFAULT_RTOL)
    atol = cfg.get("atol", DEFAULT_ATOL)
    for q_len, kv_len in cfg["shapes"]:
        kernel = build_mha_kernel(cfg, q_len, kv_len)
        Q, K, V = make_inputs(q_len, kv_len, npu_device, cfg.get("seed", 42))
        O = kernel(Q, K, V)
        torch.npu.synchronize()

        expected = ref_program(Q, K, V)
        out_f, ref_f = O.cpu().float(), expected.cpu().float()
        max_abs = (out_f - ref_f).abs().max().item()

        assert torch.isfinite(out_f).all(), f"kernel 输出包含 NaN/Inf (q_len={q_len}, kv_len={kv_len})"
        assert torch.allclose(out_f, ref_f, rtol=rtol, atol=atol), (
            f"MHA 数值超出容差: 存在元素 |out-ref| > atol + rtol*|ref| "
            f"(atol={atol:g}, rtol={rtol:g})\n"
            f"  shape: q_len={q_len}, kv_len={kv_len}, "
            f"out_dtype={cfg.get('out_dtype', 'float32')}, "
            f"dynamic={bool(cfg.get('dynamic', False))}, "
            f"num_blocks={cfg.get('num_blocks')}\n"
            f"  max_abs_diff={max_abs:.4e}"
        )
        print(
            f"[correctness] {_case_id(cfg)} q_len={q_len} kv_len={kv_len}: "
            f"max_abs={max_abs:.4e} (atol={atol:g}, rtol={rtol:g}) PASS"
        )


def _bench_desc(prof, flops: float):
    """统一性能口径: 返回 (ms, 描述), 描述以 "X.XXX ms" 开头 (3 位小数)。"""
    if hasattr(prof, "dur_ns"):
        ms = prof.dur_ns / 1e6
        return ms, f"{ms:.3f} ms | {prof.tflops(flops):.1f} TFLOPS"
    ms = float(prof)
    return ms, f"{ms:.3f} ms | {flops / (ms * 1e-3) / 1e12:.1f} TFLOPS"


@pytest.mark.parametrize("cfg", PERF_CASES, ids=PERF_IDS)
def test_mha_perf(cfg, npu_device):
    """性能用例: 平均时延 / TFLOPS, 可选对比 torch SDPA 基准 (不校验数值)。

    可用 -k "not perf" 跳过; backend 由环境变量 MHA_BENCH_BACKEND 控制
    (默认 msprof_detail); 计时参数采用 fork example_mha 口径 (30 次预热 / 500 次重复)。
    """
    backend = os.environ.get("MHA_BENCH_BACKEND", "msprof_detail")
    for q_len, kv_len in cfg["shapes"]:
        kernel = build_mha_kernel(cfg, q_len, kv_len)
        Q, K, V = make_inputs(q_len, kv_len, npu_device, cfg.get("seed", 42))
        kernel(Q, K, V)  # 预热 (首次调用包含初始化)
        torch.npu.synchronize()

        flops = 4.0 * q_len * kv_len * 128
        prof = do_bench(
            lambda: kernel(Q, K, V), backend=backend,
            _n_warmup=PERF_WARMUP, _n_repeat=PERF_REPEAT,
        )
        tl_ms, tl_desc = _bench_desc(prof, flops)
        print(f"[perf][tilelang] {_case_id(cfg)} q_len={q_len} kv_len={kv_len}: {tl_desc}")

        if os.environ.get("MHA_BENCH_TORCH", "1") != "0":
            try:
                # 预 reshape, 计时 lambda 内只含 SDPA 本身 (与 fork bench_torch_mha 一致)
                q_sdpa = Q.unsqueeze(0).unsqueeze(0)
                k_sdpa = K.unsqueeze(0).unsqueeze(0)
                v_sdpa = V.unsqueeze(0).unsqueeze(0)
                softmax_scale = 1.0 / math.sqrt(128)
                ref_prof = do_bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa, scale=softmax_scale
                    ),
                    backend=backend,
                    _n_warmup=PERF_WARMUP,
                    _n_repeat=PERF_REPEAT,
                )
                torch_ms, torch_desc = _bench_desc(ref_prof, flops)
                print(f"[perf][torch]    {_case_id(cfg)} q_len={q_len} kv_len={kv_len}: {torch_desc}")
                if torch_ms > 0 and tl_ms > 0:
                    ratio = torch_ms / tl_ms
                    verdict = f"speedup: {ratio:.2f}x" if ratio >= 1 else f"torch is {1 / ratio:.2f}x faster"
                    print(
                        f"[perf][compare] {_case_id(cfg)} q_len={q_len} kv_len={kv_len}: "
                        f"tilelang {tl_ms:.3f} ms vs torch {torch_ms:.3f} ms, {verdict}"
                    )
            except Exception as exc:  # torch 基准失败不影响 tilelang 性能结论
                print(f"[perf][torch] SDPA 基准执行失败, 跳过对比: {exc}")

        assert tl_ms > 0, "性能计时结果异常 (<= 0 ms)"
