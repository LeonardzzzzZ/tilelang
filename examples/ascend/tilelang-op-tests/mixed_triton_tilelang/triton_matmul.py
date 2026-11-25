# -*- coding: utf-8 -*-
"""triton-ascend 侧 matmul kernel 及校验入口 (混合部署验证的 triton 代表算子).

kernel 复制自 triton-ascend 官方教程 ( Ascend/triton-ascend 仓库
third_party/ascend/tutorials/03-matrix-multiplication.py, 参考 3.2.1 ):
    - fp16 输入 / fp32 累加 / fp16 输出;
    - tl.dot 分块累加 + 边界 mask;
    - Ascend 特有: extension.parallel 将每个 block 的写出切分到两个 vector core;
    - triton.autotune 两个 block 配置 (128^3 / 64^3);
    - 正确性判据与官方教程一致: atol = rtol = 1e-3 (对比 torch.matmul).

如需替换为实际的 triton_ascend 算子, 保持 run_triton_matmul_check 的
"执行 + 自校验" 接口即可, 上层混部用例无需改动.
"""

import torch
import torch_npu  # noqa: F401  # 导入即注册 npu 设备
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

DEV = "npu"


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    GROUP_SIZE_M: tl.constexpr = 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
        b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk
        a = tl.load(
            a_ptrs,
            mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # Ascend: 每个 block 的写出切分到两个 vector core (官方教程实现)
    SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 2
    for s in extension.parallel(0, 2, bind_sub_block=True):
        vec_sub_blk = extension.extract_slice(
            accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
        )
        if ACTIVATION == "leaky_relu_custom":
            vec_sub_blk = leaky_relu_custom(vec_sub_blk)
        c_sub_blk = vec_sub_blk.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c_sub_blk, mask=c_mask)


@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    """C = A @ B 的 triton 封装 (A/B: fp16, C: fp16)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


def run_triton_matmul_check(M: int = 512, N: int = 512, K: int = 512, seed: int = 0, tag: str = "") -> float:
    """执行 triton matmul 并与 torch.matmul 比对 (官方判据 atol=rtol=1e-3).

    失败抛 AssertionError, 成功返回最大绝对误差并打印结果。
    """
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16)
    b = torch.randn((K, N), device=DEV, dtype=torch.float16)
    out = matmul(a, b)
    torch.npu.synchronize()
    ref = torch.matmul(a, b)

    out_f, ref_f = out.float(), ref.float()
    max_abs = (out_f - ref_f).abs().max().item()
    assert torch.allclose(out_f, ref_f, rtol=1e-3, atol=1e-3), (
        f"triton matmul 结果超出容差: M={M}, N={N}, K={K}, max_abs={max_abs:.4e} (atol=rtol=1e-3)"
    )
    print(f"[triton-matmul{tag}] M={M} N={N} K={K} max_abs={max_abs:.4e} PASS")
    return max_abs


if __name__ == "__main__":
    run_triton_matmul_check()
