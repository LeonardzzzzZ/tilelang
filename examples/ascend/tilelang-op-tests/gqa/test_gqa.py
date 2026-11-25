# -*- coding: utf-8 -*-
"""GQA (Grouped-Query Attention) 算子 pytest 测试脚本 (自包含, 不依赖原测试工程).

来源参考 (tilelang-asc 仓库, commit 77ced9c):
    - kernel 实现:   examples/ascend/flash_attention/core.py   (flash_attention_fwd)
    - 封装与参考实现: examples/ascend/flash_attention/example_gqa.py
    - 正确性用例:    examples/ascend/flash_attention/test_gqa.py
    仓库地址: https://github.com/HelloWorld07-7/tilelang-asc/tree/77ced9c5889e1ff4b22d1f19cb77a2829195c53b/examples/ascend/flash_attention/

说明
    Q / O 使用展平布局 [S1 * G, D], K / V 为所有 query group 共享 [S2, D]。
    MHA 是 G=1 / q_len==kv_len 的特例; 本脚本聚焦 GQA 前向。

运行环境
    - Ascend NPU 服务器: CANN + torch_npu + tilelang (含 tilelang.ascend)
    - 实际测试卡为 28 个 cube 核心, 性能最优配置为 num_blocks=28 (每核一个 block)
    - 建议配合 TILELANG_DISABLE_CACHE=1 保证每次重新编译 (可选)

运行方式
    pytest gqa/test_gqa.py -v                    # 全部用例 (正确性 + 性能)
    pytest gqa/test_gqa.py -v -k correctness     # 只跑正确性
    pytest gqa/test_gqa.py -v -k "not perf"      # 跳过性能用例
    pytest gqa/test_gqa.py -v -k "s1-1024"       # 按 case id 过滤 shape
    pytest gqa/test_gqa.py -v -s                 # 显示编译/比对/计时的详细输出

kernel 对 shape 泛化的硬性约束 (不满足会在用例表中直接报错)
    1. D 固定为 128 (SIMD softmax NZ 打包路径限制);
    2. q_len = S1 * G 必须能被 block_q = 128 整除;
    3. S2 必须能被 block_kv = 128 整除;
    4. num_blocks 必须整除 M tiles 数 (q_len / 128), 本脚本默认自动求解
       (上限 28, 与实际测试卡 28 个 cube 核心一致); 若要 num_blocks=28
       满载 28 核, 需 M tiles % 28 == 0, 即 q_len % 3584 == 0;
    5. 输入 dtype 固定 bfloat16 (kernel 内部实现, fp32 累加),
       out_dtype 在 kernel 能力上支持 bfloat16 / float16 / float32。

测试规格 (通过用例默认值承载, 需变更时在单个 case 中显式指定):
    dtype = out_dtype = bfloat16 (BF16), manual_schedule = False (自动调度),
    精度容差 atol = rtol = 1.6e-2 (逐元素判据 |out - ref| <= atol + rtol * |ref|)。

环境变量
    GQA_BENCH_BACKEND    性能计时 backend, 默认 msprof_detail, 可换 wall 等
    GQA_BENCH_TORCH      性能用例是否附带 torch SDPA 基准, 默认 1 (开启)

shape 泛化用例表 (GQA_CASES) 当前为占位集合, 覆盖
    小/中/大 shape、G=1 退化、非方阵 (q_len != kv_len)、
    num_blocks=28 (28 核满载); dtype / out_dtype / manual_schedule
    按规格固定为 BF16 / BF16 / 自动调度;
    具体泛化规格确认后直接替换/扩充该表即可。
"""

import math
import os
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
    pytest.skip(f"缺少运行依赖 (tilelang / torch), 跳过 GQA 测试: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. Kernel 实现
#    逐字复制自 tilelang-asc examples/ascend/flash_attention/core.py @ 77ced9c,
#    包括 FwdTiling 与 flash_attention_fwd (含 return_lse 路径)。
#    如需同步上游 kernel 更新, 整段替换本节即可, 请勿单独改动实现细节。
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
    """Keep each Q tile in its own L0A buffer for the complete KV loop.

    Q is loaded from L1 to qk_a_l0 once per query tile. The KV loop only
    reads that buffer; PV continues to use the separate pv_a_l0 allocation.
    Both GEMMs and their L0C-to-UB drains use paired UnitFlag control 3.
    Shapes, tile sizes, arithmetic and launch geometry match the original.
    """
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
    SCALE_AFTER_MAX = math.isfinite(scale) and scale > 0

    # Implementation constraints of the SIMD softmax-packing path. These are
    # real limits of the current kernel, not tunables — validate explicitly
    # rather than advertise flexibility the code does not have.
    assert head_dim == 128, "The SIMD packing path requires head_dim == 128."
    assert BR % 2 == 0
    assert BC == 2 * VL
    assert q_len % BR == 0
    assert kv_len % BC == 0

    NUM_M_TILES = q_len // BR
    NUM_KV_BLOCKS = kv_len // BC
    NUM_BLOCKS = NUM_M_TILES if num_blocks is None else num_blocks
    assert NUM_BLOCKS > 0
    assert NUM_KV_BLOCKS > 0
    assert NUM_M_TILES % NUM_BLOCKS == 0, "M tiles must divide evenly across cores."
    M_TILES_PER_BLOCK = NUM_M_TILES // NUM_BLOCKS

    is_cast_out = out_dtype != accum_dtype
    assert out_dtype in (accum_dtype, "bfloat16", "float16"), (
        f"out_dtype must be {accum_dtype} (no cast) or a 16-bit type (bfloat16/float16); got {out_dtype!r}."
    )

    @T.macro
    def qk(kv, K, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        T.copy(K_shared[0:BC, 0:D], qk_b_l0)
        T.gemm(
            qk_a_l0,
            qk_b_l0,
            qk_acc_l0c,
            transpose_B=True,
            clear_accum=True,
            unit_flag_ctrl=3,
        )
        T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC], unit_flag_ctrl=3)

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

            old_m_reg = S.vld(m_ub[0], "NORM")

            for r in range(0, ROWS, 2):
                # A positive scale preserves ordering; scale the row maxima once.
                # This is a compile-time choice, retaining the old path otherwise.
                s00 = S.vld(S_ub[r, 0], "NORM") if SCALE_AFTER_MAX else S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full)
                s01 = S.vld(S_ub[r, VL], "NORM") if SCALE_AFTER_MAX else S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full)
                s10 = (
                    S.vld(S_ub[r + 1, 0], "NORM")
                    if SCALE_AFTER_MAX
                    else S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full)
                )
                s11 = (
                    S.vld(S_ub[r + 1, VL], "NORM")
                    if SCALE_AFTER_MAX
                    else S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full)
                )

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            block_max_scaled = (
                S.vmuls(S.vld(m_ub[0], "NORM"), T.float32(scale), softmax_full) if SCALE_AFTER_MAX else S.vld(m_ub[0], "NORM")
            )
            merged_max = S.vmax(block_max_scaled, old_m_reg, softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                # Recompute scaling instead of writing scaled scores back to UB.
                e00 = S.vexpdif(S.vmuls(p00, T.float32(scale), softmax_full), max0, softmax_full)
                e10 = S.vexpdif(S.vmuls(p10, T.float32(scale), softmax_full), max1, softmax_full)
                e01 = S.vexpdif(S.vmuls(p01, T.float32(scale), softmax_full), max0, softmax_full)
                e11 = S.vexpdif(S.vmuls(p11, T.float32(scale), softmax_full), max1, softmax_full)
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
            alpha_reg = S.vexpdif(old_m_reg, merged_max, softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            # Keep the online denominator update in one FP32 fused multiply-add.
            l_updated = S.alloc_var(accum_dtype)
            l_updated = S.vld(l_ub[0], "NORM")
            S.vmadd(l_updated, alpha_reg, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(
            pv_a_l0,
            pv_b_l0,
            pv_acc_l0c,
            transpose_B=True,
            clear_accum=True,
            unit_flag_ctrl=3,
        )
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D], unit_flag_ctrl=3)

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    updated = S.alloc_var(accum_dtype)
                    updated = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    S.vmadd(updated, alpha, cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def kv_auto(
        kv,
        K,
        V,
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
        qk(kv, K, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def kv_manual(
        kv,
        K,
        V,
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
        # [0,1,1,3,5,6,3,6,6,6,7,7].
        with T.Stage(0):
            T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        with T.Stage(1):
            T.copy(K_shared[0:BC, 0:D], qk_b_l0)
            T.gemm(
                qk_a_l0,
                qk_b_l0,
                qk_acc_l0c,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
        with T.Stage(3):
            T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC], unit_flag_ctrl=3)
        with T.Stage(5):
            softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        with T.Stage(6):
            pack_p(P_nz_ub, P_shared)
        with T.Stage(3):
            T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        with T.Stage(6):
            T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
            T.copy(V_shared[0:D, 0:BC], pv_b_l0)
            T.gemm(
                pv_a_l0,
                pv_b_l0,
                pv_acc_l0c,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
        with T.Stage(7):
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D], unit_flag_ctrl=3)
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
                        # One Q slot stays live across all KV iterations.
                        qk_a_l0: 1,
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
                    # Load Q into L0A once; qk() does not overwrite it in the KV loop.
                    T.copy(Q_shared[0:BR, 0:D], qk_a_l0)

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
                    # One Q slot stays live across all KV iterations.
                    qk_a_l0: 1,
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
                # Load Q into L0A once; qk() does not overwrite it in the KV loop.
                T.copy(Q_shared[0:BR, 0:D], qk_a_l0)

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
# 2. GQA 封装与 torch 参考实现
#    复制自 example_gqa.py @ 77ced9c。
# ---------------------------------------------------------------------------

def gqa(
    S1=8192,
    G=32,
    S2=8192,
    D_val=128,
    softmax_scale=None,
    out_dtype="bfloat16",
    num_blocks=32,
    manual_schedule=False,
):
    return flash_attention_fwd(
        q_len=S1 * G,
        kv_len=S2,
        head_dim=D_val,
        out_dtype=out_dtype,
        softmax_scale=softmax_scale,
        num_blocks=num_blocks,
        manual_schedule=manual_schedule,
    )


def ref_program(Q_flat, K, V, S1=8192, G=32, softmax_scale=None):
    """torch SDPA 参考实现 (GQA: K/V 广播到每个 query group)."""
    D_ref = Q_flat.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_ref)
    q = Q_flat.view(S1, G, D_ref).permute(1, 0, 2).unsqueeze(0)
    k = K.unsqueeze(0).unsqueeze(0)
    v = V.unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
    return out.squeeze(0).permute(1, 0, 2).contiguous().view(S1 * G, D_ref)


# ---------------------------------------------------------------------------
# 3. shape 泛化用例表
#    TODO(shape 泛化规格): 占位集合, 规格确认后直接替换/扩充本表。
#    字段说明 (dtype / out_dtype / manual_schedule 由默认值承载测试规格:
#    BF16 / BF16 / 自动调度, 需变更时在单个 case 中显式覆盖):
#        S1 / G / S2        GQA 三个维度; D 恒为 128 (可省略, 默认 128,
#                            传其他值会被约束校验拦截)
#        dtype              输入类型, 默认 bfloat16 (kernel 仅实现 bf16 输入)
#        out_dtype          输出类型, 默认 bfloat16; 规格固定 BF16,
#                            kernel 能力上可选 float16 / float32
#        softmax_scale      缩放系数, 默认 None (即 1/sqrt(D))
#        num_blocks         核数划分, 默认 None (自动求解: 不超过 28 的最大因数);
#                            28 核满载最优配置为 28, 要求 q_len % 3584 == 0
#        manual_schedule    是否使用固定 stage 的手工调度, 默认 False; 规格固定
#                            自动调度 (kernel 能力上 True 需 S2 >= 1024)
#        rtol               相对容差, 默认 1.6e-2 (逐元素判据 |out-ref| <= atol + rtol*|ref|)
#        atol               绝对容差, 默认 1.6e-2
#        seed               随机数据种子, 默认 42
#        perf               是否纳入性能用例, 默认 False
# ---------------------------------------------------------------------------

BLOCK_Q = 128       # FwdTiling.block_q, q_len 必须被其整除
BLOCK_KV = 128      # FwdTiling.block_kv, S2 必须被其整除
NUM_BLOCKS_CAP = 28  # 自动求解 num_blocks 的上限: 实际测试卡为 28 个 cube 核心
DEFAULT_RTOL = 1.6e-2  # 精度规格: 逐元素判据 |out-ref| <= atol + rtol*|ref| 的相对容差
DEFAULT_ATOL = 1.6e-2  # 精度规格: 绝对容差

GQA_CASES = [
    # -- 冒烟: 最小 shape (G=1 退化为 MHA 形态; S2=128 为 KV block 下界,
    #    若在服务器上触发流水线最小迭代约束可调大 S2) --
    dict(S1=128, G=1, S2=128),
    # -- 小 shape --
    dict(S1=128, G=2, S2=256),
    dict(S1=256, G=2, S2=512),
    # -- 中 shape --
    dict(S1=1024, G=8, S2=1024),
    dict(S1=512, G=4, S2=1024),                      # 非方阵: q_len(2048) != kv_len(1024)
    # -- num_blocks=28 专项 (匹配 28 cube 核心, 每核一个 block, 性能最优配置):
    #    要求 M tiles % 28 == 0, 即 q_len = S1*G 必须被 3584 整除 --
    dict(S1=128, G=28, S2=1024, num_blocks=28),               # 最小 28 核 case: 1 M tile/block
    dict(S1=896, G=32, S2=2048, num_blocks=28),               # 中等: 8 M tiles/block
    dict(S1=8192, G=28, S2=8192, num_blocks=28, perf=True),   # 大: 64 M tiles/block, 28 核满载基准
    dict(S1=7168, G=32, S2=8192, num_blocks=28, perf=True),   # 大: G=32 语义 shape, q_len 同为 229376 (M tiles=1792)
    dict(S1=7168, G=32, S2=7168, num_blocks=28, perf=True),   # 大: 方形 shape (S1=S2=7168, 56 个 KV block)
    # -- 仓库原始基准 shape (M tiles=2048 不含因子 28, 显式固定仓库的 num_blocks=32
    #    用于对齐仓库数据; 32 > 28 会跨核二次调度, 在 28 核卡上非最优) --
    dict(S1=8192, G=32, S2=8192, num_blocks=32, perf=True),
]


# ---------------------------------------------------------------------------
# 4. 用例校验 / num_blocks 求解 / kernel 编译缓存
# ---------------------------------------------------------------------------

def _case_id(cfg: dict) -> str:
    """生成可读且可用于 pytest -k 过滤的 case id (小写)."""
    parts = [f"S1-{cfg['S1']}", f"G-{cfg['G']}", f"S2-{cfg['S2']}"]
    if cfg.get("out_dtype", "bfloat16") != "bfloat16":
        parts.append(str(cfg["out_dtype"]))
    if cfg.get("manual_schedule", False):
        parts.append("manual")
    return "_".join(parts).lower()


def resolve_num_blocks(cfg: dict) -> int:
    """求解满足 'num_blocks 整除 M tiles (q_len/128)' 的核数。

    cfg 显式指定 num_blocks 时优先使用; 否则取不超过 cap 的最大因数。
    """
    if cfg.get("num_blocks"):
        return int(cfg["num_blocks"])
    m_tiles = (cfg["S1"] * cfg["G"]) // BLOCK_Q
    cap = int(cfg.get("num_blocks_cap", NUM_BLOCKS_CAP))
    for d in range(min(cap, m_tiles), 0, -1):
        if m_tiles % d == 0:
            return d
    return 1


def validate_case(cfg: dict) -> None:
    """静态校验 shape 配置满足 kernel 硬性约束, 不满足时给出中文报错."""
    tag = _case_id(cfg)
    S1, G, S2, D = cfg["S1"], cfg["G"], cfg["S2"], cfg.get("D", 128)
    q_len = S1 * G

    if D != 128:
        raise ValueError(f"[{tag}] SIMD softmax 打包路径要求 D=128, 当前 D={D}")
    if q_len <= 0 or S2 <= 0:
        raise ValueError(f"[{tag}] S1*G 与 S2 必须为正数, 当前 q_len={q_len}, S2={S2}")
    if q_len % BLOCK_Q != 0:
        raise ValueError(f"[{tag}] q_len=S1*G={q_len} 必须能被 block_q={BLOCK_Q} 整除")
    if S2 % BLOCK_KV != 0:
        raise ValueError(f"[{tag}] S2={S2} 必须能被 block_kv={BLOCK_KV} 整除")

    nb = resolve_num_blocks(cfg)
    m_tiles = q_len // BLOCK_Q
    if m_tiles % nb != 0:
        raise ValueError(f"[{tag}] num_blocks={nb} 必须整除 M tiles 数 q_len/{BLOCK_Q}={m_tiles}")
    if cfg.get("manual_schedule", False) and S2 < 8 * BLOCK_KV:
        raise ValueError(
            f"[{tag}] manual_schedule=True 要求 S2 >= 8*{BLOCK_KV}={8 * BLOCK_KV} (固定 stage 最大为 7)"
        )
    if cfg.get("out_dtype", "bfloat16") not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"[{tag}] out_dtype 仅支持 bfloat16/float16/float32, 当前 {cfg.get('out_dtype')!r}")


# 由用例表派生的 pytest 参数化数据 (需放在上面 helper 函数定义之后)
CASE_IDS = [_case_id(c) for c in GQA_CASES]
PERF_CASES = [c for c in GQA_CASES if c.get("perf")]
PERF_IDS = [_case_id(c) for c in PERF_CASES]


_KERNEL_CACHE = {}


def build_gqa_kernel(cfg: dict):
    """编译 (并缓存) 指定配置的 GQA kernel, 编译方式与仓库 example 一致."""
    key = (
        cfg["S1"], cfg["G"], cfg["S2"], cfg.get("D", 128),
        cfg.get("out_dtype", "bfloat16"),
        cfg.get("softmax_scale"),
        resolve_num_blocks(cfg),
        bool(cfg.get("manual_schedule", False)),
    )
    if key not in _KERNEL_CACHE:
        validate_case(cfg)
        print(f"[compile] {_case_id(cfg)} (num_blocks={key[6]}) ...")
        func = gqa(
            S1=cfg["S1"],
            G=cfg["G"],
            S2=cfg["S2"],
            D_val=cfg.get("D", 128),
            softmax_scale=cfg.get("softmax_scale"),
            out_dtype=cfg.get("out_dtype", "bfloat16"),
            num_blocks=resolve_num_blocks(cfg),
            manual_schedule=bool(cfg.get("manual_schedule", False)),
        )
        _KERNEL_CACHE[key] = tilelang.compile(
            func,
            out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _KERNEL_CACHE[key]


TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def make_inputs(cfg: dict, device):
    """按配置生成随机输入 (dtype 取自 case 的 dtype 字段, 默认 bfloat16; 种子固定可复现)."""
    S1, G, S2, D = cfg["S1"], cfg["G"], cfg["S2"], cfg.get("D", 128)
    dtype = cfg.get("dtype", "bfloat16")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"不支持的输入 dtype: {dtype!r}, 可选: {sorted(TORCH_DTYPES)}")
    tdtype = TORCH_DTYPES[dtype]
    torch.manual_seed(cfg.get("seed", 42))
    Q = torch.randn(S1 * G, D, dtype=tdtype, device=device)
    K = torch.randn(S2, D, dtype=tdtype, device=device)
    V = torch.randn(S2, D, dtype=tdtype, device=device)
    return Q, K, V


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  # 导入即注册 npu 设备
    except Exception:
        pass
    return hasattr(torch, "npu") and torch.npu.is_available()


# ---------------------------------------------------------------------------
# 5. pytest 用例
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npu_device():
    if not _npu_available():
        pytest.skip("未检测到可用 NPU (需要 torch_npu + CANN 环境)")
    return torch.device("npu")


def test_case_configs_satisfy_kernel_constraints():
    """静态校验 GQA_CASES 用例表满足 kernel 硬性约束 (无需 NPU, 秒级)."""
    assert GQA_CASES, "GQA_CASES 用例表不能为空"
    for cfg in GQA_CASES:
        validate_case(cfg)


@pytest.mark.parametrize("cfg", GQA_CASES, ids=CASE_IDS)
def test_gqa_correctness(cfg, npu_device):
    """正确性用例: tilelang kernel 输出 vs torch SDPA 参考 (输入 dtype 取自 case, 默认 bf16)。

    判据 (逐元素): |out - ref| <= atol + rtol * |ref|, 精度规格 atol = rtol = 1.6e-2,
    可在 case 中通过 atol / rtol 字段覆盖。
    """
    kernel = build_gqa_kernel(cfg)
    S1, G, S2, D = cfg["S1"], cfg["G"], cfg["S2"], cfg.get("D", 128)

    Q, K, V = make_inputs(cfg, npu_device)
    O = kernel(Q, K, V)
    torch.npu.synchronize()

    expected = ref_program(Q, K, V, S1=S1, G=G, softmax_scale=cfg.get("softmax_scale"))
    out_f = O.cpu().float()
    ref_f = expected.cpu().float()

    assert torch.isfinite(out_f).all(), "kernel 输出包含 NaN/Inf"

    max_abs = (out_f - ref_f).abs().max().item()
    ref_max = ref_f.abs().max().item()
    rel = max_abs / ref_max if ref_max > 0 else max_abs
    rtol = cfg.get("rtol", DEFAULT_RTOL)
    atol = cfg.get("atol", DEFAULT_ATOL)

    ok = torch.allclose(out_f, ref_f, rtol=rtol, atol=atol)
    print(
        f"[correctness] {_case_id(cfg)}: max_abs={max_abs:.4e}, "
        f"rel={rel:.4e} (atol={atol:g}, rtol={rtol:g})"
    )
    assert ok, (
        f"GQA 数值超出容差: 存在元素 |out-ref| > atol + rtol*|ref| "
        f"(atol={atol:g}, rtol={rtol:g})\n"
        f"  shape: S1={S1}, G={G}, S2={S2}, D={D}, "
        f"out_dtype={cfg.get('out_dtype', 'bfloat16')}, "
        f"manual_schedule={cfg.get('manual_schedule', False)}\n"
        f"  max_abs_diff={max_abs:.4e}, ref_max={ref_max:.4e}, rel={rel:.4e}"
    )


def _bench_desc(prof, flops: float):
    """统一性能口径: 返回 (ms, 描述), 描述以 "X.XXX ms" 开头 (3 位小数)。"""
    if hasattr(prof, "dur_ns"):
        ms = prof.dur_ns / 1e6
        return ms, f"{ms:.3f} ms | {prof.tflops(flops):.1f} TFLOPS"
    ms = float(prof)
    return ms, f"{ms:.3f} ms | {flops / (ms * 1e-3) / 1e12:.1f} TFLOPS"


@pytest.mark.parametrize("cfg", PERF_CASES, ids=PERF_IDS)
def test_gqa_perf(cfg, npu_device):
    """性能用例: 平均时延 / TFLOPS, 可选对比 torch SDPA 基准 (不校验数值)。

    可用 -k "not perf" 跳过; backend 由环境变量 GQA_BENCH_BACKEND 控制 (默认 msprof_detail)。
    """
    kernel = build_gqa_kernel(cfg)
    S1, G, S2, D = cfg["S1"], cfg["G"], cfg["S2"], cfg.get("D", 128)

    Q, K, V = make_inputs(cfg, npu_device)
    kernel(Q, K, V)  # 预热 (首次调用包含初始化)
    torch.npu.synchronize()

    backend = os.environ.get("GQA_BENCH_BACKEND", "msprof_detail")
    flops = 4.0 * (S1 * G) * S2 * D

    prof = do_bench(lambda: kernel(Q, K, V), backend=backend, _n_warmup=10, _n_repeat=50)
    tl_ms, tl_desc = _bench_desc(prof, flops)
    print(f"[perf][tilelang] {_case_id(cfg)}: {tl_desc}")

    if os.environ.get("GQA_BENCH_TORCH", "1") != "0":
        try:
            softmax_scale = cfg.get("softmax_scale") or 1.0 / math.sqrt(D)
            q_sdpa = Q.view(S1, G, D).permute(1, 0, 2).unsqueeze(0).contiguous()
            k_sdpa = K.unsqueeze(0).unsqueeze(0)
            v_sdpa = V.unsqueeze(0).unsqueeze(0)
            ref_prof = do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, scale=softmax_scale
                ),
                backend=backend,
                _n_warmup=10,
                _n_repeat=50,
            )
            torch_ms, torch_desc = _bench_desc(ref_prof, flops)
            print(f"[perf][torch]    {_case_id(cfg)}: {torch_desc}")
            if torch_ms > 0 and tl_ms > 0:
                ratio = torch_ms / tl_ms
                verdict = f"speedup: {ratio:.2f}x" if ratio >= 1 else f"torch is {1 / ratio:.2f}x faster"
                print(f"[perf][compare] {_case_id(cfg)}: tilelang {tl_ms:.3f} ms vs torch {torch_ms:.3f} ms, {verdict}")
        except Exception as exc:  # torch 基准失败不影响 tilelang 性能结论
            print(f"[perf][torch] SDPA 基准执行失败, 跳过对比: {exc}")

    assert tl_ms > 0, "性能计时结果异常 (<= 0 ms)"
