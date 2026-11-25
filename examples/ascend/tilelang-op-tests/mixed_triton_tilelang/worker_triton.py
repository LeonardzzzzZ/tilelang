# -*- coding: utf-8 -*-
"""跨进程混测 worker: 仅执行 triton matmul kernel 并自校验.

用法 (独立运行):
    python mixed_triton_tilelang/worker_triton.py

约定 (供 test_mixed_deployment.py 检查):
    - 成功: stdout 含 "WORKER_PASS", 退出码 0
    - 失败: stdout 含 "WORKER_FAIL", 退出码 1
"""

import os
import sys
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from triton_matmul import run_triton_matmul_check


def main() -> int:
    try:
        max_abs = run_triton_matmul_check(tag="[worker-triton]")
    except Exception as exc:
        traceback.print_exc()
        print(f"[worker-triton] WORKER_FAIL: {exc}")
        return 1
    print(f"[worker-triton] WORKER_PASS matmul(512x512x512 fp16) max_abs={max_abs:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
