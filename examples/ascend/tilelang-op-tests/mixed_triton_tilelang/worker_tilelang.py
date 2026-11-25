# -*- coding: utf-8 -*-
"""跨进程混测 worker: 仅执行 tilelang GQA kernel 并自校验.

用法 (独立运行):
    python mixed_triton_tilelang/worker_tilelang.py

约定 (供 test_mixed_deployment.py 检查):
    - 成功: stdout 含 "WORKER_PASS", 退出码 0
    - 失败: stdout 含 "WORKER_FAIL", 退出码 1

复用 gqa/test_gqa.py 中的 kernel 与判据 (BF16, atol=rtol=1.6e-2),
case 为混测专用小 shape (编译更快)。
"""

import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch_npu  # noqa: F401  # 导入即注册 npu 设备

from gqa.test_gqa import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    build_gqa_kernel,
    make_inputs,
    ref_program,
)

GQA_CASE = dict(S1=128, G=2, S2=256)  # q_len=256, M tiles=2, 自动 num_blocks=2


def main() -> int:
    device = torch.device("npu")
    try:
        kernel = build_gqa_kernel(GQA_CASE)
        Q, K, V = make_inputs(GQA_CASE, device)
        O = kernel(Q, K, V)
        torch.npu.synchronize()
        expected = ref_program(Q, K, V, S1=GQA_CASE["S1"], G=GQA_CASE["G"])

        out_f, ref_f = O.cpu().float(), expected.cpu().float()
        max_abs = (out_f - ref_f).abs().max().item()
        if not torch.isfinite(out_f).all():
            print(f"[worker-tilelang] WORKER_FAIL: 输出包含 NaN/Inf")
            return 1
        if not torch.allclose(out_f, ref_f, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
            print(
                f"[worker-tilelang] WORKER_FAIL: max_abs={max_abs:.4e} "
                f"超出容差 (atol={DEFAULT_ATOL:g}, rtol={DEFAULT_RTOL:g})"
            )
            return 1
    except Exception as exc:
        traceback.print_exc()
        print(f"[worker-tilelang] WORKER_FAIL: {exc}")
        return 1
    print(f"[worker-tilelang] WORKER_PASS GQA(S1=128,G=2,S2=256) max_abs={max_abs:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
