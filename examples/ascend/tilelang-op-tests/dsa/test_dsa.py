# -*- coding: utf-8 -*-
"""DSA (GLM DSA, DeepSeek Sparse Attention) 算子 pytest 测试脚本 (自包含, 不依赖原测试工程).

来源参考 (HhpPCc007/tilelang-asc, commit 3c34da6):
    - kernel 实现:   examples/ascend/glm_dsa/core.py             (glm_dsa_fwd)
    - 封装与参考实现: examples/ascend/glm_dsa/example_glm_dsa.py  (make_inputs / reference_attention)
    - 正确性用例:    examples/ascend/glm_dsa/test_glm_dsa.py
    仓库地址: https://github.com/HhpPCc007/tilelang-asc/blob/3c34da6ee7bd102f91b85f2fa46bdc5e71b0b96a/examples/ascend/glm_dsa/core.py

输入契约 (kernel 固定属性: 64 头 / head_dim=256 / scale=0.0625 / TopK 上限 2048 / dropout 0)
    Q          [B, 64, sq, 256]  bfloat16
    K / V      [B, 64, T,  256]  bfloat16 (每头独立)
    indices    [B, sq, Ksel]     int32, Ksel = min(2048, T),
                 每条 query 的索引为 [0, T) 内唯一位置, 无 -1 填充
    mask       [B, 1, sq, T]     bool (调用时转 uint8 连续视图), 非零 = 参与注意力
    O          [B, sq, 64, 256]  bfloat16 (连续)

运行环境
    - Ascend NPU 服务器: CANN + torch_npu + tilelang (tilelang.ascend, 需含 tvm.tirx)
    - 实际测试卡为 28 个 cube 核心 (kernel 默认 num_blocks 自动取 min(核数, 任务数))
    - 建议配合 TILELANG_DISABLE_CACHE=1 保证每次重新编译 (可选)

运行方式
    pytest dsa/test_dsa.py -v -s                    # 全部用例
    pytest dsa/test_dsa.py -v -k "not perf"         # 跳过性能用例
    pytest dsa/test_dsa.py -v -k "t257"             # 按 case id / shape 过滤
    pytest dsa/test_dsa.py -v -k "vector_mask"      # 只跑 SIMD mask 探针

shape 约束 (不满足会在用例表中直接报错; 约束极宽松)
    1. B / sq / T 均只需为正数 (非对齐长度走标量回退路径, 如 T=96/160/257/2049);
    2. Ksel = min(2048, T) 由 kv_len 推导, 用例无需指定;
    3. heads_per_task 可选 (1,2,4,8,16,32,64), 默认 4;
    4. num_blocks 显式指定时必须满足 0 < num_blocks <= B*sq*(64/heads_per_task),
       默认 None = 自动取 min(cube 核数, 任务数);
    5. mask_mode 取 mixed (随机 ~90% 保留) / all_keep / all_drop。

测试规格 (通过用例默认值承载, 需变更时在单个 case 中显式指定)
    dtype 与 out_dtype 均为 bfloat16 (kernel 内部实现), FP32 累加;
    精度容差 atol = rtol = 1e-2 (fork test/example 口径), 判据为逐元素
    |out - ref| <= atol + rtol * |ref|, 参考 = FP32 per-query golden。

环境变量
    DSA_BENCH_BACKEND    性能计时 backend, 默认 msprof_detail

shape 泛化用例表 (DSA_CASES) 当前为 fork test_glm_dsa.py 的 14 个 attention 回归
    全量移植 (单/多 TopK block、batch、全 mask、残块尾、非对齐 T、自定义 head
    分组、任务数不整除核数) + README 性能参考 shape;
    具体泛化规格确认后直接替换/扩充该表即可。
"""

import os
from dataclasses import dataclass

import pytest

try:
    import torch
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S
    from tilelang.carver.arch.driver import get_num_cube_cores
    from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_k_layout
    from tilelang.profiler import do_bench
    from tvm import tirx
except ImportError as exc:  # pragma: no cover - 仅在依赖缺失的环境触发
    pytest.skip(f"缺少运行依赖 (tilelang / torch / tvm), 跳过 DSA 测试: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. Kernel 实现
#    逐字复制自 HhpPCc007/tilelang-asc examples/ascend/glm_dsa/core.py @ 3c34da6,
#    包括常量 / FwdTiling / _copy_row_pair / _expand_mask_row / _vector_mask_penalty
#    / glm_dsa_fwd / pick_num_blocks。
#    如需同步上游 kernel 更新, 整段替换本节即可, 请勿单独改动实现细节。
# ---------------------------------------------------------------------------

NUM_HEADS = 64
HEAD_DIM = 256
SOFTMAX_SCALE = 0.0625
INDEX_TOPK = 2048

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"

# Replaces a masked score outright.  exp(MASK_VALUE - row_max) underflows to
# exactly zero, so a masked lane contributes nothing to either the softmax sum
# or the PV GEMM.
MASK_VALUE = -3.402823e38

# Floor applied to a row maximum.  A row whose lanes are all masked would give
# exp(0) = 1 for every lane; flooring the maximum keeps them underflowing to 0.
MAX_FLOOR = -1.0e30


@dataclass(frozen=True)
class FwdTiling:
    # The SIMD probability packing path is built around a 128-wide index tile.
    block_topk: int = 128
    heads_per_task: int = 4


def _copy_row_pair(dst, src, row_stride):
    """Gather two rows in one GM->UB DMA whose source rows are ``row_stride`` apart.

    Only an ascending pair is expressible: the first burst lands on the smaller
    KV position and the second on the larger one, regardless of the order the two
    indices appear in ``indices``.  ``mask_penalty`` reproduces that pairing.
    """
    row_bytes = HEAD_DIM * 2
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        2,
        row_bytes,
        0,
        0,
        0,
        0,
        row_stride * row_bytes,
        row_bytes,
    )


@T.macro
def _expand_mask_row(mask_row_ub, mask_words_ub, kv_len):
    """Widen the raw uint8 mask row to uint32 using only vector loads/stores."""
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full32 = S.pset(32, "PAT_ALL")
        for chunk in T.serial(kv_len // 256):
            raw = S.vld(mask_row_ub[chunk * 256], "NORM")
            packed = T.reinterpret(raw, "uint32x64")
            byte_mask = S.vdup(T.uint32(255), "uint32", full32)
            byte0 = S.vand(packed, byte_mask, full32)
            byte1 = S.vand(S.vshrs(packed, 8, full32), byte_mask, full32)
            byte2 = S.vand(S.vshrs(packed, 16, full32), byte_mask, full32)
            byte3 = S.vand(S.vshrs(packed, 24, full32), byte_mask, full32)
            even_low, even_high = S.vintlv(byte0, byte2)
            odd_low, odd_high = S.vintlv(byte1, byte3)
            S.vst2(mask_words_ub[chunk * 256], even_low, odd_low)
            S.vst2(mask_words_ub[chunk * 256 + 128], even_high, odd_high)


@T.macro
def _vector_mask_penalty(indices_ub, mask_words_ub, penalty_ub, kv_len):
    """Produce this tile's 128 flags, matching the ascending paired K/V DMA."""
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full = S.pset(32, "PAT_ALL")
        even, odd = S.vld2(indices_ub[0], "DINTLV_B32")
        first = T.reinterpret(S.vmin(even, odd, full), "uint32x64")
        second = T.reinterpret(S.vmax(even, odd, full), "uint32x64")
        keep_first = S.vgather2(T.access_ptr(mask_words_ub[0], "r", extent=kv_len), first, full)
        keep_second = S.vgather2(T.access_ptr(mask_words_ub[0], "r", extent=kv_len), second, full)
        zero = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
        one = S.vdup(T.float32(1.0), ACCUM_DTYPE, full)
        first_flag = S.vsel(zero, one, S.vcmps(keep_first, T.uint32(0), full, op="ne"))
        second_flag = S.vsel(zero, one, S.vcmps(keep_second, T.uint32(0), full, op="ne"))
        S.vst2(penalty_ub[0], first_flag, second_flag)


def glm_dsa_fwd(
    batch_size,
    seq_len,
    kv_len,
    top_k,
    *,
    num_blocks=None,
    tiling=None,
):
    tiling = tiling or FwdTiling()

    # Capture plain compile-time scalars in the hygienic T.macros below.
    BLOCK_TOP_K = tiling.block_topk
    HEADS_PER_TASK = tiling.heads_per_task
    assert HEADS_PER_TASK in (1, 2, 4, 8, 16, 32, 64)
    HEAD_GROUPS = NUM_HEADS // HEADS_PER_TASK
    CHUNK_ROWS = 16
    PIPELINE_STAGES = 2
    M_PAD = 16  # Cube M granularity; every row carries the same query vector.
    ROWS_PER_AIV = M_PAD // 2
    KV_ROWS_PER_AIV = BLOCK_TOP_K // 2
    VECTOR_LANES = 64  # Ascend fp32 SIMD lane count

    assert batch_size > 0 and seq_len > 0 and kv_len > 0, "batch_size, seq_len and kv_len must be positive"
    assert top_k == min(INDEX_TOPK, kv_len), "top_k must be min(index_topk, kv_len)"
    # The softmax handles two fp32x64 vectors and packs one bf16x128 vector.
    assert BLOCK_TOP_K == 128, "The current SIMD softmax packing requires block_topk == 128"

    # One attention_mask row costs kv_len bytes of UB and its DMA source origin
    # is the row start, so bulk staging is only legal when that origin and the
    # row length are both 32B aligned and the row fits the window.  Any other
    # kv_len reads the mask scalar from GM instead: same result, slower pass.
    MASK_STAGE_BYTES = 32 * 1024
    MASK_STAGED = (kv_len % 32 == 0) and (kv_len <= MASK_STAGE_BYTES)
    # Bound the widened row's UB footprint; other lengths keep the scalar path.
    VECTOR_MASK = MASK_STAGED and kv_len % 256 == 0 and kv_len <= 8192 and top_k % BLOCK_TOP_K == 0
    # K/V always read their indices from GM, independent of the mask path.

    NUM_BLOCKS = pick_num_blocks(seq_len, batch_size, heads_per_task=HEADS_PER_TASK) if num_blocks is None else num_blocks
    assert NUM_BLOCKS > 0, "num_blocks must be positive"
    NUM_TASKS = batch_size * seq_len * HEAD_GROUPS
    assert NUM_BLOCKS <= NUM_TASKS, "num_blocks exceeds the number of query-head tasks"
    NUM_TOPK_BLOCKS = (top_k + BLOCK_TOP_K - 1) // BLOCK_TOP_K

    # -- shared T.macros --------------------------------------------------

    def read_index(Indices, batch_id, query_id, block_base, lane):
        """Read a sparse index directly from GM."""
        return T.int32(Indices[batch_id, query_id, block_base + lane])

    def read_keep(Mask, mask_row_ub, batch_id, query_id, position):
        """Whether a KV position participates, staged in UB or read scalar."""
        if MASK_STAGED:
            return mask_row_ub[position] != T.uint8(0)
        return Mask[batch_id, 0, query_id, position] != T.uint8(0)

    @T.macro
    def gather_rows(src, batch_id, query_id, head, block_base, valid_count, sid, indices_src, raw_ub, nz_ub, l1):
        """Each AIV gathers one half of the sparse tile.

        Lanes at or past ``valid_count`` re-gather the last valid position rather
        than being skipped.  Their probabilities are zero, but the row still
        feeds the PV MMA: an unwritten row comes from whatever the buffer held
        before -- uninitialised UB on the first tile -- and one Inf/NaN there
        turns ``0 * V`` into NaN and poisons the whole output.  Repeating a real
        index keeps every L1 row finite, and ``mask_penalty`` masks those lanes
        out by position.
        """
        for chunk in T.Serial(KV_ROWS_PER_AIV // CHUNK_ROWS):
            row_start = sid * KV_ROWS_PER_AIV + chunk * CHUNK_ROWS
            for pair in T.Serial(CHUNK_ROWS // 2):
                lane0 = T.min(row_start + 2 * pair, valid_count - 1)
                lane1 = T.min(row_start + 2 * pair + 1, valid_count - 1)
                index0 = read_index(indices_src, batch_id, query_id, block_base, lane0)
                index1 = read_index(indices_src, batch_id, query_id, block_base, lane1)
                T.assume(index0 >= 0)
                T.assume(index0 < kv_len)
                T.assume(index1 >= 0)
                T.assume(index1 < kv_len)
                if index0 == index1:
                    T.copy(src[batch_id, head, index0, :], raw_ub[2 * pair, :])
                    T.copy(src[batch_id, head, index1, :], raw_ub[2 * pair + 1, :])
                else:
                    first_index = T.min(index0, index1)
                    row_stride = T.max(index0, index1) - first_index
                    _copy_row_pair(
                        T.access_ptr(raw_ub[2 * pair, 0], "w"),
                        T.access_ptr(src[batch_id, head, first_index, 0], "r"),
                        row_stride,
                    )

            T.copy(raw_ub, nz_ub[:CHUNK_ROWS, :])
            T.copy(nz_ub[:CHUNK_ROWS, :], l1[row_start : row_start + CHUNK_ROWS, :])

    @T.macro
    def mask_penalty(Mask, mask_row_ub, penalty_ub, valid_count, batch_id, query_id, block_base, indices_src):
        """Fold attention_mask and the partial-tile tail into one keep flag.

        ``penalty_ub[i]`` becomes 0.0 for a participating slot and 1.0
        otherwise; ``softmax_pack`` turns that into a lane predicate.  The mask
        is indexed by KV position. This fallback keeps scalar lookups; aligned
        full-tile lengths within the UB budget use _vector_mask_penalty instead.

        Every AIV covers the whole ``BLOCK_TOP_K`` block: the dual copy splits
        ``qk_acc_l0c`` along M only, so each AIV's ``S_ub`` rows still span all
        ``BLOCK_TOP_K`` KV columns.  The gather divides the rows between the two
        AIVs, but the flag vector must not -- a half-filled vector would mask
        the other AIV's columns as well as its own.

        The lane order must match ``gather_rows``: a paired DMA lands the smaller
        KV position in ``lane0``, so the mask is looked up through the same
        min/max pairing.  Looking it up by the raw ``indices`` order would drop
        the wrong lane whenever two adjacent indices are not ascending.  Lanes at
        or past ``valid_count`` are never written here, which leaves their
        penalty at 1.0 -- they are padding rows repeating a real KV position and
        must stay out of the softmax.
        """
        for lane in T.Serial(BLOCK_TOP_K):
            penalty_ub[lane] = T.float32(1.0)
        for chunk in T.Serial(BLOCK_TOP_K // CHUNK_ROWS):
            row_start = chunk * CHUNK_ROWS
            valid_rows = T.max(T.min(valid_count - row_start, CHUNK_ROWS), 0)
            for pair in T.Serial(valid_rows // 2):
                lane0 = row_start + 2 * pair
                index0 = read_index(indices_src, batch_id, query_id, block_base, lane0)
                index1 = read_index(indices_src, batch_id, query_id, block_base, lane0 + 1)
                index0 = T.max(T.min(index0, kv_len - 1), 0)
                index1 = T.max(T.min(index1, kv_len - 1), 0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, T.min(index0, index1)):
                    penalty_ub[lane0] = T.float32(0.0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, T.max(index0, index1)):
                    penalty_ub[lane0 + 1] = T.float32(0.0)
            if valid_rows % 2 != 0:
                tail_lane = row_start + valid_rows - 1
                tail_index = read_index(indices_src, batch_id, query_id, block_base, tail_lane)
                tail_index = T.max(T.min(tail_index, kv_len - 1), 0)
                if read_keep(Mask, mask_row_ub, batch_id, query_id, tail_index):
                    penalty_ub[tail_lane] = T.float32(0.0)

    @T.macro
    def softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, penalty_ub):
        with T.SimdVF(latency=735):
            S.mem_bar("VST_VLD")
            fp32_mask = S.pset(32, "PAT_ALL")
            scalar_mask = S.pset(32, "PAT_VL1")
            bf16_mask = S.pset(16, "PAT_VL128")
            mask_value = S.vdup(T.float32(MASK_VALUE), ACCUM_DTYPE, fp32_mask)
            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS_PER_AIV + 1, BLOCK_TOP_K),
                DTYPE,
            )
            nz_stride = T.int32(((ROWS_PER_AIV + 1) << 16) | 1)
            keep_lo = S.vcmps(S.vld(penalty_ub[0], "NORM"), T.float32(0.5), fp32_mask, op="lt")
            keep_hi = S.vcmps(S.vld(penalty_ub[VECTOR_LANES], "NORM"), T.float32(0.5), fp32_mask, op="lt")

            for row in range(ROWS_PER_AIV):
                scores_lo = S.vmuls(S.vld(S_ub[row, 0], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                scores_hi = S.vmuls(S.vld(S_ub[row, VECTOR_LANES], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask)
                masked_lo = S.vsel(scores_lo, mask_value, keep_lo)
                masked_hi = S.vsel(scores_hi, mask_value, keep_hi)
                S.vsts(S_ub[row, 0], masked_lo, fp32_mask, "NORM_B32")
                S.vsts(S_ub[row, VECTOR_LANES], masked_hi, fp32_mask, "NORM_B32")
                max_acc = S.vmax(masked_lo, masked_hi, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # Online softmax: m_new = max(m_old, block_max).
                    old_max = S.vld(m_ub[row], "BRC_B32")
                    S.vsts(old_m_ub[row], old_max, scalar_mask, "ONEPT_B32")
                    max_acc = S.vmax(max_acc, old_max, fp32_mask)
                row_max = S.vcmax(max_acc, fp32_mask)
                row_max = S.vmaxs(row_max, T.float32(MAX_FLOOR), scalar_mask)
                S.vsts(m_ub[row], row_max, scalar_mask, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            # Pack unnormalized P = exp(score - m_new) directly in BF16 NZ.
            for row in range(ROWS_PER_AIV):
                row_max = S.vld(m_ub[row], "BRC_B32")
                even_scores, odd_scores = S.vld2(S_ub[row, 0], "DINTLV_B32")
                even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
                odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
                sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
                even_bf16 = S.vcvt(even_prob, DTYPE, fp32_mask, sat=False, part=0)
                odd_bf16 = S.vcvt(odd_prob, DTYPE, fp32_mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(even_bf16, "uint16x128"),
                    T.reinterpret(odd_bf16, "uint16x128"),
                    bf16_mask,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    bf16_mask,
                    update=True,
                )
                prob_sum = S.vcadd(sum_acc, fp32_mask)
                if NUM_TOPK_BLOCKS > 1:
                    # alpha = exp(m_old - m_new); l_new = alpha * l_old + sum(P).
                    alpha = S.vexpdif(S.vld(old_m_ub[row], "BRC_B32"), row_max, fp32_mask)
                    S.vsts(alpha_ub[row], alpha, scalar_mask, "ONEPT_B32")
                    denominator = S.vadd(
                        S.vmul(S.vld(l_ub[row], "BRC_B32"), alpha, fp32_mask),
                        prob_sum,
                        scalar_mask,
                    )
                    S.vsts(l_ub[row], denominator, scalar_mask, "ONEPT_B32")
                else:
                    S.vsts(l_ub[row], prob_sum, scalar_mask, "ONEPT_B32")

    @T.macro
    def init_softmax_state(m_ub, l_ub):
        with T.SimdVF(latency=200):
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            initial_max = S.vdup(T.float32(MAX_FLOOR), ACCUM_DTYPE, full)
            initial_sum = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
            for row in range(ROWS_PER_AIV):
                S.vsts(m_ub[row], initial_max, one, "ONEPT_B32")
                S.vsts(l_ub[row], initial_sum, one, "ONEPT_B32")

    @T.macro
    def clear_output(O_ub):
        with T.SimdVF(latency=142):
            full = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), ACCUM_DTYPE, full)
            for row in range(ROWS_PER_AIV):
                for column in range(0, HEAD_DIM, VECTOR_LANES):
                    S.vsts(O_ub[row, column], zero, full, "NORM_B32")

    @T.macro
    def accumulate_output(O_ub, O_tmp_ub, alpha_ub):
        # O = alpha * O + P@V.  At the first tile alpha is exp(MAX_FLOOR - m),
        # which underflows to 0, so the cleared accumulator needs no special case.
        with T.SimdVF(latency=318):
            full = S.pset(32, "PAT_ALL")
            for row in range(ROWS_PER_AIV):
                alpha = S.vld(alpha_ub[row], "BRC_B32")
                for column in range(0, HEAD_DIM, VECTOR_LANES):
                    prev = S.vld(O_ub[row, column], "NORM")
                    cur = S.vld(O_tmp_ub[row, column], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, full), cur, full)
                    S.vsts(O_ub[row, column], updated, full, "NORM_B32")

    @T.macro
    def finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id, head, sid):
        # Normalize in FP32, convert to BF16, then store the single live row.
        # A fully masked query has l == 0; clamp the denominator so it emits
        # zeros instead of dividing by zero.
        with T.SimdVF(latency=814):
            full = S.pset(32, "PAT_ALL")
            denominator = S.vld(l_ub[0], "BRC_B32")
            safe_denominator = S.vmaxs(denominator, T.float32(1.0e-30), full)
            for column in range(0, HEAD_DIM, VECTOR_LANES):
                normalized = S.vdiv(S.vld(O_ub[0, column], "NORM"), safe_denominator, full)
                packed = S.vcvt(normalized, DTYPE, full, sat=False)
                S.vsts(O_cast_ub[0, column], packed, full, "PK_B32")
        # Every M row is a copy of the same query, so row 0 holds the same values
        # on both AIVs.  Split the store by column rather than guarding it with
        # ``sid == 0``: InsertSync rejects a sid-guarded MTE3 because both AIV
        # programs then carry a write to the same region (AIV0.MTE3 -> AIV1.MTE3,
        # distance 0).  Disjoint column halves need no cross-AIV ordering.
        half = HEAD_DIM // 2
        T.copy(
            O_cast_ub[0, sid * half : (sid + 1) * half],
            O[batch_id, query_id, head, sid * half : (sid + 1) * half],
        )

    # -- auto-scheduled Cube + Vector kernel ------------------------------

    @T.prim_func
    def main(
        Q: T.Buffer((batch_size, NUM_HEADS, seq_len, HEAD_DIM), DTYPE),
        K: T.Buffer((batch_size, NUM_HEADS, kv_len, HEAD_DIM), DTYPE),
        V: T.Buffer((batch_size, NUM_HEADS, kv_len, HEAD_DIM), DTYPE),
        Indices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        Mask: T.Buffer((batch_size, 1, seq_len, kv_len), "uint8"),
        O: T.Buffer((batch_size, seq_len, NUM_HEADS, HEAD_DIM), DTYPE),
    ):
        with T.MixedKernel(NUM_BLOCKS) as (core_id, sid):
            Q_l1 = T.alloc_l1((M_PAD, HEAD_DIM), DTYPE)
            K_l1 = T.alloc_l1((BLOCK_TOP_K, HEAD_DIM), DTYPE)
            V_l1 = T.alloc_l1((BLOCK_TOP_K, HEAD_DIM), DTYPE)
            P_l1 = T.alloc_l1((M_PAD, BLOCK_TOP_K), DTYPE)
            # The L1 GEMM path is NT-only, so PV stages both operands in L0 and
            # transposes V on the L1->L0 copy: B is (N=HEAD_DIM, K=BLOCK_TOP_K).
            pv_a_l0 = T.alloc_l0a((M_PAD, BLOCK_TOP_K), DTYPE)
            pv_b_l0 = T.alloc_l0b((HEAD_DIM, BLOCK_TOP_K), DTYPE)
            qk_acc_l0c = T.alloc_l0c((M_PAD, BLOCK_TOP_K), ACCUM_DTYPE)
            pv_acc_l0c = T.alloc_l0c((M_PAD, HEAD_DIM), ACCUM_DTYPE)

            Q_ub = T.alloc_shared((M_PAD, HEAD_DIM), DTYPE)
            K_ub = T.alloc_shared((CHUNK_ROWS, HEAD_DIM), DTYPE)
            K_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, HEAD_DIM), DTYPE)
            V_ub = T.alloc_shared((CHUNK_ROWS, HEAD_DIM), DTYPE)
            V_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, HEAD_DIM), DTYPE)
            S_ub = T.alloc_shared((ROWS_PER_AIV, BLOCK_TOP_K), ACCUM_DTYPE)
            P_nz_ub = T.alloc_shared((ROWS_PER_AIV + 1, BLOCK_TOP_K), DTYPE)
            O_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), ACCUM_DTYPE)
            O_tmp_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), ACCUM_DTYPE)
            # Same bytes as O_ub would do, but a separate buffer keeps the
            # in-place compaction trick out of the correctness path.
            O_cast_ub = T.alloc_shared((ROWS_PER_AIV, HEAD_DIM), DTYPE)
            # Per-row running max m, exp sum l, and output rescaling factor alpha.
            m_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            l_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            alpha_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            old_m_ub = T.alloc_shared((ROWS_PER_AIV,), ACCUM_DTYPE)
            penalty_ub = T.alloc_shared((BLOCK_TOP_K,), ACCUM_DTYPE)
            mask_row_ub = T.alloc_shared((kv_len if MASK_STAGED else 32,), "uint8")
            mask_words_ub = T.alloc_shared((kv_len if VECTOR_MASK else 1,), "uint32")
            mask_indices_ub = T.alloc_shared((BLOCK_TOP_K,), "int32")

            T.annotate_layout(
                {
                    # Cube operand views in L1 must be MajorK: the compact-NZ UB
                    # staging is only accepted into a canonical NZ destination.
                    Q_l1: make_ascend_major_k_layout(Q_l1),
                    K_l1: make_ascend_major_k_layout(K_l1),
                    V_l1: make_ascend_major_k_layout(V_l1),
                    K_nz_ub: make_ascend_compact_nz_layout(K_nz_ub),
                    V_nz_ub: make_ascend_compact_nz_layout(V_nz_ub),
                    P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    K_l1: 2,
                    V_l1: 2,
                    P_l1: 2,
                    pv_a_l0: 2,
                    # A full HEAD_DIM x BLOCK_TOP_K tile already fills L0B.
                    pv_b_l0: 1,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 2,
                    K_ub: 2,
                    K_nz_ub: 2,
                    V_ub: 2,
                    V_nz_ub: 2,
                    P_nz_ub: 2,
                    O_tmp_ub: 2,
                    alpha_ub: 2,
                    # Loop-carried across tiles: must stay single-versioned.
                    O_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    old_m_ub: 1,
                }
            )

            if VECTOR_MASK:
                T.annotate_buffer_versions({mask_words_ub: 1, mask_indices_ub: 2})

            # Each core gets exactly its share, including the partial final round.
            for task in T.serial((NUM_TASKS + NUM_BLOCKS - 1 - core_id) // NUM_BLOCKS):
                query_group = task * NUM_BLOCKS + core_id
                query = query_group // HEAD_GROUPS
                head_start = (query_group % HEAD_GROUPS) * HEADS_PER_TASK
                batch_id = query // seq_len
                query_id = query % seq_len
                # The mask row is per query and per KV position, so when it can
                # be staged it is loaded once per query/head-group task and reused by all
                # index tiles.  Otherwise every lookup stays a GM scalar read.
                if MASK_STAGED:
                    T.copy(Mask[batch_id, 0, query_id, 0:kv_len], mask_row_ub)

                if VECTOR_MASK:
                    _expand_mask_row(mask_row_ub, mask_words_ub, kv_len)

                for head_offset in T.serial(HEADS_PER_TASK):
                    head = head_start + head_offset
                    # Every padded M row carries the same query vector, so all
                    # 16 Cube rows -- and both AIVs -- compute the same result.
                    # GM->UB row copies followed by one UB->L1 copy keep the
                    # ND2NZ conversion on a full 16-row tile.
                    # AIV0 owns the full Q tile; two writers would overlap in L1.
                    if sid == 0:
                        for row in range(M_PAD):
                            T.copy(Q[batch_id, head, query_id, :], Q_ub[row, :])
                        T.copy(Q_ub, Q_l1)
                    if NUM_TOPK_BLOCKS > 1:
                        init_softmax_state(m_ub, l_ub)
                        clear_output(O_ub)

                    for block in T.Pipelined(
                        NUM_TOPK_BLOCKS,
                        num_stages=PIPELINE_STAGES,
                        annotations={"enable_offset": True},
                    ):
                        block_base = block * BLOCK_TOP_K
                        block_valid = T.min(top_k - block_base, BLOCK_TOP_K)

                        # QK -> masked sparse softmax -> PV.
                        gather_rows(
                            K,
                            batch_id,
                            query_id,
                            head,
                            block_base,
                            block_valid,
                            sid,
                            Indices,
                            K_ub,
                            K_nz_ub,
                            K_l1,
                        )
                        # The L1 GEMM uses hidden L0 scratch shared with PV.
                        with T.Cube():
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.gemm(Q_l1, K_l1, qk_acc_l0c, transpose_B=True, clear_accum=True)
                            T.ascend_pipe_barrier("PIPE_ALL")
                        T.dual_copy(qk_acc_l0c, S_ub)

                        # Fetch V while the softmax runs; K and V are independent.
                        gather_rows(
                            V,
                            batch_id,
                            query_id,
                            head,
                            block_base,
                            block_valid,
                            sid,
                            Indices,
                            V_ub,
                            V_nz_ub,
                            V_l1,
                        )

                        if VECTOR_MASK:
                            T.copy(Indices[batch_id, query_id, block_base : block_base + BLOCK_TOP_K], mask_indices_ub)
                            _vector_mask_penalty(mask_indices_ub, mask_words_ub, penalty_ub, kv_len)
                        else:
                            mask_penalty(
                                Mask,
                                mask_row_ub,
                                penalty_ub,
                                block_valid,
                                batch_id,
                                query_id,
                                block_base,
                                Indices,
                            )
                        softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, penalty_ub)
                        T.dual_copy(P_nz_ub[:ROWS_PER_AIV, :], P_l1)
                        # Keep PV operand loads and GEMM in one Cube task.
                        with T.Cube():
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.copy(P_l1, pv_a_l0)
                            T.copy(V_l1, pv_b_l0, transpose=True)
                            T.ascend_pipe_barrier("PIPE_ALL")
                            T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
                            T.ascend_pipe_barrier("PIPE_ALL")

                        if NUM_TOPK_BLOCKS > 1:
                            T.dual_copy(pv_acc_l0c, O_tmp_ub)
                            accumulate_output(O_ub, O_tmp_ub, alpha_ub)
                        else:
                            T.dual_copy(pv_acc_l0c, O_ub)

                    finalize_output(O_ub, O_cast_ub, l_ub, O, batch_id, query_id, head, sid)

    return main


def pick_num_blocks(seq_len, batch_size=1, cores=None, *, heads_per_task=None):
    heads_per_task = FwdTiling().heads_per_task if heads_per_task is None else heads_per_task
    assert heads_per_task in (1, 2, 4, 8, 16, 32, 64)
    assert batch_size > 0 and seq_len > 0
    cores = get_num_cube_cores(torch.npu.current_device()) if cores is None else cores
    assert cores > 0, "cores must be positive"
    tasks = batch_size * seq_len * (NUM_HEADS // heads_per_task)
    return min(cores, tasks)


# ---------------------------------------------------------------------------
# 2. 输入生成与 torch 参考实现
#    复制自 fork example_glm_dsa.py, 增加 seed / mask_mode 参数 (规格由用例承载)。
# ---------------------------------------------------------------------------

def make_inputs(cfg: dict, device):
    """按配置生成随机输入 (fork make_inputs; mask_mode 控制 mask 形态)."""
    batch_size, seq_len, kv_len = cfg["batch_size"], cfg["seq_len"], cfg["kv_len"]
    torch.manual_seed(cfg.get("seed", 88888888))
    top_k = min(INDEX_TOPK, kv_len)
    q = torch.randn((batch_size, NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.bfloat16, device=device)
    k = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device=device)
    v = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device=device)
    # Dense top-k, exactly what the caller produces:
    #   topk = min(index_topk, index_scores.shape[-1])
    #   ind = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
    index_scores = torch.rand(batch_size, seq_len, kv_len, device=device)
    indices = index_scores.topk(top_k, dim=-1).indices.to(torch.int32)
    mask_mode = cfg.get("mask_mode", "mixed")
    if mask_mode == "all_keep":
        mask = torch.ones((batch_size, 1, seq_len, kv_len), dtype=torch.bool, device=device)
    elif mask_mode == "all_drop":
        mask = torch.zeros((batch_size, 1, seq_len, kv_len), dtype=torch.bool, device=device)
    else:  # mixed: 随机保留 ~90% (fork make_inputs 口径)
        mask = torch.rand(batch_size, 1, seq_len, kv_len, device=device) > 0.1
    return q, k, v, indices, mask


def reference_attention(q, k, v, indices, mask):
    """FP32 per-query golden (fork reference_attention, Python 循环, 大 shape 较慢)."""
    batch_size, heads, seq_len, dim = q.shape
    mask_u8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    out = torch.empty((batch_size, seq_len, heads, dim), dtype=torch.float32, device=q.device)
    for batch in range(batch_size):
        k_batch = k[batch]
        v_batch = v[batch]
        for query in range(seq_len):
            # torch.gather/advanced indexing require int64 indices.
            idx = indices[batch, query].long()
            gathered_k = k_batch[:, idx, :].float()  # [H, Ksel, D]
            gathered_v = v_batch[:, idx, :].float()
            scores = torch.einsum("hd,hkd->hk", q[batch, :, query, :].float(), gathered_k) * SOFTMAX_SCALE
            keep = mask_u8[batch, 0, query][idx].ne(0)  # [Ksel]
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            # Softmax over an all-masked row is NaN; the kernel emits zeros there.
            probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[batch, query] = torch.einsum("hk,hkd->hd", probabilities, gathered_v)
    return out


# ---------------------------------------------------------------------------
# 3. shape 泛化用例表
#    TODO(shape 泛化规格): 当前为 fork test_glm_dsa.py 的 14 个 attention 回归
#    全量移植 + README 性能参考 shape, 规格确认后直接替换/扩充本表。
#    字段说明 (dtype / out_dtype / 容差由默认值承载测试规格, 需变更时在
#    单个 case 中显式覆盖):
#        batch_size / seq_len / kv_len   三个 shape 维度 (只需为正数,
#                                       非对齐长度走 kernel 标量回退路径);
#                                       Ksel = min(2048, kv_len) 自动推导
#        num_blocks        核数划分, 默认 None (自动: min(cube 核数, 任务数));
#                          显式指定须满足 0 < nb <= B*sq*(64/heads_per_task)
#        heads_per_task    每任务头数, 默认 4, 可选 (1,2,4,8,16,32,64)
#        mask_mode         mixed (默认, 随机 ~90% 保留) / all_keep / all_drop
#        rtol / atol       精度容差, 默认 1e-2 (fork test 口径)
#        seed              随机数据种子, 默认 88888888 (fork make_inputs 口径)
#        perf              是否纳入性能用例, 默认 False
# ---------------------------------------------------------------------------

DEFAULT_RTOL = 1e-2  # 精度规格: 逐元素判据 |out-ref| <= atol + rtol*|ref|
DEFAULT_ATOL = 1e-2  # 精度规格: 绝对容差
PERF_WARMUP = 30     # 性能计时预热次数 (fork example 口径)
PERF_REPEAT = 100    # 性能计时重复次数 (fork example 口径)

DSA_CASES = [
    # -- 单 TopK block (Ksel <= 128, 无 online-softmax 更新), T=96 走 GM 标量 mask --
    dict(batch_size=1, seq_len=32, kv_len=96, num_blocks=8),
    dict(batch_size=1, seq_len=32, kv_len=96, num_blocks=8, mask_mode="all_keep"),
    # -- 多 TopK block (running max/sum + 输出缩放) --
    dict(batch_size=1, seq_len=32, kv_len=1024, num_blocks=8),
    dict(batch_size=1, seq_len=16, kv_len=2048, num_blocks=8, perf=True),
    # -- batch 维 + query 数多于核数 --
    dict(batch_size=2, seq_len=32, kv_len=512, num_blocks=16),
    # -- 全 mask: 输出应为 0 而非 NaN --
    dict(batch_size=1, seq_len=32, kv_len=512, num_blocks=8, mask_mode="all_drop"),
    # -- 尾部残块 (Ksel=160 非 128 倍数) --
    dict(batch_size=1, seq_len=16, kv_len=160, num_blocks=8),
    # -- kv_len 非 32 对齐 / 超出 32KB 暂存窗口 (mask 走 GM 标量路径);
    #    T=2049 同时触达 2048 的 TopK 上限 --
    dict(batch_size=1, seq_len=8, kv_len=257, num_blocks=8),
    dict(batch_size=1, seq_len=4, kv_len=2049, num_blocks=4),
    # -- 自定义 head 分组 (num_blocks 自动 = min(核数, 任务数)) --
    dict(batch_size=1, seq_len=16, kv_len=2048, heads_per_task=32),
    dict(batch_size=1, seq_len=17, kv_len=2049, heads_per_task=2),
    dict(batch_size=1, seq_len=1, kv_len=257, heads_per_task=64),
    # -- 任务数不整除核数 (28 核留部分轮次) --
    dict(batch_size=1, seq_len=3, kv_len=257, num_blocks=28),
    # -- README 性能参考 shape --
    dict(batch_size=1, seq_len=448, kv_len=4096, perf=True),
]


# ---------------------------------------------------------------------------
# 4. 用例校验 / kernel 编译缓存
# ---------------------------------------------------------------------------

def _case_id(cfg: dict) -> str:
    """生成可读且可用于 pytest -k 过滤的 case id (小写)."""
    parts = [f"b{cfg['batch_size']}", f"sq{cfg['seq_len']}", f"t{cfg['kv_len']}"]
    if cfg.get("num_blocks"):
        parts.append(f"nb{cfg['num_blocks']}")
    if cfg.get("heads_per_task", 4) != 4:
        parts.append(f"hpt{cfg['heads_per_task']}")
    if cfg.get("mask_mode", "mixed") != "mixed":
        parts.append(cfg["mask_mode"])
    return ("dsa_" + "_".join(parts)).lower()


def validate_case(cfg: dict) -> None:
    """静态校验 shape 配置满足 kernel 硬性约束, 不满足时给出中文报错."""
    tag = _case_id(cfg)
    batch_size, seq_len, kv_len = cfg["batch_size"], cfg["seq_len"], cfg["kv_len"]
    if min(batch_size, seq_len, kv_len) <= 0:
        raise ValueError(f"[{tag}] batch_size/seq_len/kv_len 必须为正, 当前 ({batch_size}, {seq_len}, {kv_len})")

    heads_per_task = cfg.get("heads_per_task", 4)
    if heads_per_task not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"[{tag}] heads_per_task={heads_per_task} 不在支持集合 (1,2,4,8,16,32,64)")
    num_tasks = batch_size * seq_len * (NUM_HEADS // heads_per_task)

    nb = cfg.get("num_blocks")
    if nb is not None and not (0 < nb <= num_tasks):
        raise ValueError(
            f"[{tag}] 显式 num_blocks={nb} 必须满足 0 < num_blocks <= 任务数 {num_tasks} "
            f"(B*sq*(64/heads_per_task))"
        )
    if cfg.get("mask_mode", "mixed") not in ("mixed", "all_keep", "all_drop"):
        raise ValueError(f"[{tag}] mask_mode 仅支持 mixed/all_keep/all_drop, 当前 {cfg.get('mask_mode')!r}")


_KERNEL_CACHE = {}


def build_dsa_kernel(cfg: dict):
    """编译 (并缓存) 指定配置的 DSA kernel, 编译方式与 fork example 一致 (target=ascend)."""
    key = (
        cfg["batch_size"], cfg["seq_len"], cfg["kv_len"],
        cfg.get("num_blocks"), cfg.get("heads_per_task", 4),
    )
    if key not in _KERNEL_CACHE:
        validate_case(cfg)
        heads_per_task = cfg.get("heads_per_task", 4)
        tiling = None if heads_per_task == 4 else FwdTiling(heads_per_task=heads_per_task)
        nb = cfg.get("num_blocks")
        print(
            f"[compile] {_case_id(cfg)} (top_k={min(INDEX_TOPK, cfg['kv_len'])}, "
            f"heads/task={heads_per_task}, num_blocks={'auto' if nb is None else nb}) ...",
        )
        func = glm_dsa_fwd(
            cfg["batch_size"], cfg["seq_len"], cfg["kv_len"], min(INDEX_TOPK, cfg["kv_len"]),
            num_blocks=nb, tiling=tiling,
        )
        _KERNEL_CACHE[key] = tilelang.compile(
            func,
            target="ascend",
            out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _KERNEL_CACHE[key]


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  # 导入即注册 npu 设备
    except Exception:
        pass
    return hasattr(torch, "npu") and torch.npu.is_available()


# 由用例表派生的 pytest 参数化数据 (需放在上面 helper 函数定义之后)
CASE_IDS = [_case_id(c) for c in DSA_CASES]
PERF_CASES = [c for c in DSA_CASES if c.get("perf")]
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
    """静态校验 DSA_CASES 用例表满足 kernel 硬性约束 (无需 NPU, 秒级)."""
    assert DSA_CASES, "DSA_CASES 用例表不能为空"
    for cfg in DSA_CASES:
        validate_case(cfg)


@pytest.mark.parametrize("cfg", DSA_CASES, ids=CASE_IDS)
def test_dsa_correctness(cfg, npu_device):
    """正确性用例: tilelang kernel 输出 (BF16) vs FP32 per-query golden。

    判据 (逐元素): |out - ref| <= atol + rtol * |ref|, 默认 atol = rtol = 1e-2
    (fork test 口径); mask_mode=all_drop 时额外校验输出全零 (kernel 契约:
    全 mask 查询输出 0 而非 NaN)。
    注意: reference_attention 按 query 逐行 Python 循环, 大 shape (如 sq=448)
    生成 golden 较慢, 属预期行为。
    """
    kernel = build_dsa_kernel(cfg)
    q, k, v, indices, mask = make_inputs(cfg, npu_device)

    expected = reference_attention(q, k, v, indices, mask)
    output = kernel(q, k, v, indices, mask.view(torch.uint8))
    torch.npu.synchronize()

    assert output.dtype == torch.bfloat16, f"输出 dtype 应为 bfloat16, 当前 {output.dtype}"
    assert tuple(output.shape) == (cfg["batch_size"], cfg["seq_len"], NUM_HEADS, HEAD_DIM)
    assert output.is_contiguous(), "输出应为连续张量"

    out_f, ref_f = output.float(), expected
    rtol = cfg.get("rtol", DEFAULT_RTOL)
    atol = cfg.get("atol", DEFAULT_ATOL)

    if cfg.get("mask_mode") == "all_drop":
        nonzero = torch.count_nonzero(out_f).item()
        assert nonzero == 0, f"全 mask 查询应输出全零, 实际非零元素 {nonzero} 个 (不应出现 NaN)"

    max_abs = (out_f - ref_f).abs().max().item()
    assert torch.allclose(out_f, ref_f, rtol=rtol, atol=atol), (
        f"DSA 数值超出容差: 存在元素 |out-ref| > atol + rtol*|ref| "
        f"(atol={atol:g}, rtol={rtol:g})\n"
        f"  shape: B={cfg['batch_size']}, sq={cfg['seq_len']}, T={cfg['kv_len']}, "
        f"Ksel={min(INDEX_TOPK, cfg['kv_len'])}, "
        f"num_blocks={cfg.get('num_blocks')}, "
        f"heads_per_task={cfg.get('heads_per_task', 4)}, "
        f"mask_mode={cfg.get('mask_mode', 'mixed')}\n"
        f"  max_abs_diff={max_abs:.4e}"
    )
    print(
        f"[correctness] {_case_id(cfg)}: max_abs={max_abs:.4e} "
        f"(atol={atol:g}, rtol={rtol:g}) PASS"
    )


def test_both_halves_of_a_tile_contribute(npu_device):
    """单个 tile 的两半 AIV 列都必须参与 (fork 同名回归)。

    Q/K 全零使 128 个选中位置同权, V 前半 0 后半 1, 期望输出为均值 0.5;
    若 keep-flag 向量只覆盖单个 AIV 收集的一半, 输出会退化为 0 (或 1)。
    """
    batch_size, seq_len, kv_len, num_blocks = 1, 8, 128, 8
    cfg = dict(batch_size=batch_size, seq_len=seq_len, kv_len=kv_len, num_blocks=num_blocks)
    kernel = build_dsa_kernel(cfg)

    q = torch.zeros(batch_size, NUM_HEADS, seq_len, HEAD_DIM, dtype=torch.bfloat16, device=npu_device)
    k = torch.zeros(batch_size, NUM_HEADS, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=npu_device)
    v = torch.zeros(batch_size, NUM_HEADS, kv_len, HEAD_DIM, dtype=torch.bfloat16, device=npu_device)
    v[:, :, 64:, :] = 1.0
    indices = (
        torch.arange(kv_len, dtype=torch.int32, device=npu_device)
        .view(1, 1, kv_len)
        .expand(batch_size, seq_len, kv_len)
        .contiguous()
    )
    attn_mask = torch.ones(batch_size, 1, seq_len, kv_len, dtype=torch.uint8, device=npu_device)

    actual = kernel(q, k, v, indices, attn_mask)
    torch.npu.synchronize()

    expected = torch.full_like(actual.float(), 0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    print("[correctness] dsa_both_halves: 均值 0.5 PASS")


def _build_mask_flag_probe(kv_len):
    """SIMD mask flag 探针 kernel (fork test_glm_dsa 同款, 复用 kernel 的两个宏)."""

    @T.prim_func
    def flags(Mask: T.Buffer((kv_len,), "uint8"), Indices: T.Buffer((128,), "int32"), Out: T.Buffer((128,), "float32")):
        with T.Kernel(1) as _:
            row = T.alloc_shared((kv_len,), "uint8")
            words = T.alloc_shared((kv_len,), "uint32")
            selected = T.alloc_shared((128,), "int32")
            penalty = T.alloc_shared((128,), "float32")
            T.copy(Mask, row)
            T.copy(Indices, selected)
            _expand_mask_row(row, words, kv_len)
            _vector_mask_penalty(selected, words, penalty, kv_len)
            T.copy(penalty, Out)

    return flags


@pytest.mark.parametrize("kv_len", [256, 2048, 8192])
def test_vector_mask_flags(kv_len, npu_device):
    """SIMD 探针: 精确 0/1 flag, 乱序配对 + 多次调用换 mask (fork 同名回归)."""
    print(f"[vector-mask] inputs: mask=[{kv_len}] uint8, indices=[128] int32", flush=True)
    kernel = tilelang.compile(_build_mask_flag_probe(kv_len), target="ascend", out_idx=-1)
    generator = torch.Generator().manual_seed(20261106)
    indices = torch.randperm(kv_len, generator=generator)[:128].int()
    mixed = torch.randint(0, 4, (kv_len,), generator=generator, dtype=torch.uint8)
    for step, mask in enumerate((mixed, torch.ones_like(mixed), torch.zeros_like(mixed), mixed.roll(13))):
        selected = (indices + 17 * step) % kv_len
        ordered = selected.reshape(64, 2).sort(dim=1).values.reshape(128).long()
        expected = (mask[ordered] == 0).float()
        actual = kernel(mask.to(npu_device), selected.to(npu_device))
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print(f"[vector-mask] kv_len={kv_len}: 4 组 mask flag 全部精确匹配 PASS")


def _bench_desc(prof, flops: float):
    """统一性能口径: 返回 (ms, 描述), 描述以 "X.XXX ms" 开头 (3 位小数)。"""
    if hasattr(prof, "dur_ns"):
        ms = prof.dur_ns / 1e6
        return ms, f"{ms:.3f} ms | {prof.tflops(flops):.1f} TFLOPS"
    ms = float(prof)
    return ms, f"{ms:.3f} ms | {flops / (ms * 1e-3) / 1e12:.1f} TFLOPS"


@pytest.mark.parametrize("cfg", PERF_CASES, ids=PERF_IDS)
def test_dsa_perf(cfg, npu_device):
    """性能用例: 平均时延 / TFLOPS (不校验数值, 正确性见 correctness 用例)。

    计时口径与 fork example 一致 (msprof_detail, 30 预热 / 100 重复),
    编译与 golden 均不计入被测 lambda; DSA 无直接等价的 torch 算子,
    不附带 torch 基准 (fork 以 GBSA 为对照, 不在本工程内)。
    可用 -k "not perf" 跳过; backend 由环境变量 DSA_BENCH_BACKEND 控制。
    """
    kernel = build_dsa_kernel(cfg)
    q, k, v, indices, mask = make_inputs(cfg, npu_device)
    mask_u8 = mask.view(torch.uint8)

    kernel(q, k, v, indices, mask_u8)  # 预热 (首次调用包含初始化)
    torch.npu.synchronize()

    backend = os.environ.get("DSA_BENCH_BACKEND", "msprof_detail")
    flops = 4.0 * cfg["batch_size"] * cfg["seq_len"] * NUM_HEADS * HEAD_DIM * min(INDEX_TOPK, cfg["kv_len"])
    prof = do_bench(
        lambda: kernel(q, k, v, indices, mask_u8),
        backend=backend,
        _n_warmup=PERF_WARMUP,
        _n_repeat=PERF_REPEAT,
    )
    tl_ms, tl_desc = _bench_desc(prof, flops)
    print(f"[perf][tilelang] {_case_id(cfg)}: {tl_desc}")
    assert tl_ms > 0, "性能计时结果异常 (<= 0 ms)"
