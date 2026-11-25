# -*- coding: utf-8 -*-
"""triton-ascend 与 tilelang 混合部署验证 (功能正确性; 本轮不含性能对比).

目的
    验证 tilelang 算子 (GQA) 与 triton 算子 (matmul) 在如下部署形态下
    均能正确执行、互不干扰:
      A. 同进程: 两个框架在同一 Python 进程内先后执行 (双向顺序) 与交替执行;
         另以子进程验证双框架的双向导入顺序;
      B. 跨进程: 两个独立 worker 进程在同一张 NPU 卡上
         顺序执行 (双向顺序) 与并发执行。

文件
    test_mixed_deployment.py   pytest 入口 (本文件)
    triton_matmul.py           triton 侧 matmul kernel (官方教程 03 复制) + 校验入口
    worker_tilelang.py         独立进程 worker: 仅执行 tilelang GQA 并自校验
    worker_triton.py           独立进程 worker: 仅执行 triton matmul 并自校验

用法
    pytest mixed_triton_tilelang/test_mixed_deployment.py -v -s            # 全部场景
    pytest mixed_triton_tilelang/test_mixed_deployment.py -v -k same_process
    pytest mixed_triton_tilelang/test_mixed_deployment.py -v -k import_order
    pytest mixed_triton_tilelang/test_mixed_deployment.py -v -k cross_process

环境 (服务器)
    - Ascend NPU + CANN + torch_npu
    - tilelang (tilelang.ascend)                  —— GQA kernel 侧
    - triton-ascend (pip install triton-ascend, 参考 3.2.1 / CANN 9.0.0) —— matmul 侧
    - pytest
    - 依赖缺失时整个模块 skip, 原因逐项列出缺失模块 (pytest -rs 查看)

环境变量
    MIXED_WORKER_TIMEOUT   跨进程 worker 超时时间 (秒), 默认 900 (编译慢可调大)

判定口径
    - tilelang GQA:  BF16, 逐元素 |out-ref| <= atol + rtol*|ref|, atol=rtol=1.6e-2;
    - triton matmul: fp16, 官方教程口径 atol=rtol=1e-3;
    - 跨进程: worker 退出码为 0 且 stdout 含 WORKER_PASS 标记。
"""

import importlib
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

# ---------------------------------------------------------------------------
# 依赖逐项导入: 缺失时 skip 原因会精确列出缺失模块与安装提示 (pytest -rs 查看)
# ---------------------------------------------------------------------------
_missing = []


def _import_or_record(name: str, hint: str = ""):
    """导入模块, 失败时记录到 _missing (用于生成精确的 skip 原因), 返回模块或 None."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        entry = f"{name}  ({type(exc).__name__}: {exc})"
        if hint:
            entry += f"  <- {hint}"
        _missing.append(entry)
        return None


torch = _import_or_record("torch")
_import_or_record("torch_npu", "需 CANN 环境 (source /usr/local/Ascend/ascend-toolkit/set_env.sh)")
triton = _import_or_record("triton", "triton-ascend: pip install triton-ascend")
if triton is not None:
    _import_or_record(
        "triton.language.extra.cann.extension",
        "triton-ascend 3.2.x 参考版含该模块; 社区版 triton 或旧版本无此模块",
    )
tilelang = _import_or_record("tilelang")
if tilelang is not None:
    _import_or_record("tilelang.ascend.language", "tilelang 的 ascend 后端子模块, 检查 tilelang 安装")

if not _missing:
    # 仓库内模块 (依赖外部包全部就绪后再导入, 避免二次报错淹没根因)
    try:
        from triton_matmul import run_triton_matmul_check  # noqa: F401
    except ImportError as exc:
        _missing.append(f"triton_matmul (本仓库 mixed_triton_tilelang/ 内模块)  ({exc})")
    try:
        from gqa.test_gqa import (  # noqa: F401
            DEFAULT_ATOL,
            DEFAULT_RTOL,
            build_gqa_kernel,
            make_inputs,
            ref_program,
            validate_case,
        )
    except ImportError as exc:
        _missing.append(f"gqa.test_gqa (本仓库 gqa/ 内模块)  ({exc})")

if _missing:
    pytest.skip(
        "缺少运行依赖, 跳过混合部署测试, 缺失清单:\n  - "
        + "\n  - ".join(_missing)
        + "\n  (逐项可用 python -c 'import 模块名' 复现验证)",
        allow_module_level=True,
    )

# 混测使用的 GQA 小 shape case (编译更快; q_len=256, M tiles=2, 自动 num_blocks=2)
GQA_MIX_CASE = dict(S1=128, G=2, S2=256)
validate_case(GQA_MIX_CASE)

WORKER_TIMEOUT = int(os.environ.get("MIXED_WORKER_TIMEOUT", 900))
WORKERS = ("worker_tilelang.py", "worker_triton.py")
WORKER_PASS_MARK = "WORKER_PASS"


# ---------------------------------------------------------------------------
# 公共执行/校验函数
# ---------------------------------------------------------------------------

def _run_gqa_check(tag: str = "") -> float:
    """执行 tilelang GQA (小 case) 并按 GQA 规格校验, 失败抛断言, 成功返回最大误差.

    build_gqa_kernel 在 gqa 模块内有编译缓存, 同一进程内多个用例共享一次编译。
    """
    device = torch.device("npu")
    kernel = build_gqa_kernel(GQA_MIX_CASE)
    Q, K, V = make_inputs(GQA_MIX_CASE, device)
    O = kernel(Q, K, V)
    torch.npu.synchronize()

    expected = ref_program(Q, K, V, S1=GQA_MIX_CASE["S1"], G=GQA_MIX_CASE["G"])
    out_f, ref_f = O.cpu().float(), expected.cpu().float()
    max_abs = (out_f - ref_f).abs().max().item()

    assert torch.isfinite(out_f).all(), "tilelang GQA 输出包含 NaN/Inf"
    assert torch.allclose(out_f, ref_f, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL), (
        f"tilelang GQA 结果超出容差: max_abs={max_abs:.4e} "
        f"(atol={DEFAULT_ATOL:g}, rtol={DEFAULT_RTOL:g})"
    )
    print(f"[tilelang-gqa{tag}] S1=128 G=2 S2=256 max_abs={max_abs:.4e} PASS")
    return max_abs


def _worker_fail_message(name, returncode, stdout, stderr):
    return (
        f"worker {name} 失败 (退出码 {returncode}):\n"
        f"--- stdout ---\n{stdout}\n--- stderr (末尾 2000 字符) ---\n{stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def npu():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("未检测到可用 NPU (需要 torch_npu + CANN 环境)")
    return torch.device("npu")


# ---------------------------------------------------------------------------
# 场景 A: 同进程
# ---------------------------------------------------------------------------

def test_env_info(npu):
    """信息性用例: 打印双框架版本, 便于服务器日志追溯 (配合 -s 查看)."""
    print(
        f"[env] torch={torch.__version__} "
        f"tilelang={getattr(tilelang, '__version__', 'unknown')} "
        f"triton={getattr(triton, '__version__', 'unknown')}"
    )


def test_same_process_import_order():
    """双框架双向导入顺序验证 (子进程执行, 避免本进程导入顺序已固化)."""
    cases = [
        ("triton-first", "import torch, torch_npu, triton; import gqa.test_gqa"),
        ("tilelang-first", "import torch, torch_npu; import gqa.test_gqa; import triton"),
    ]
    for label, code in cases:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"导入顺序 {label} 失败:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
        print(f"[import-order] {label}: OK")


def test_same_process_tilelang_then_triton(npu):
    """同进程: 先执行 tilelang GQA, 再执行 triton matmul, 双方结果均需正确."""
    _run_gqa_check(tag="[tilelang-then-triton]")
    run_triton_matmul_check(tag="[tilelang-then-triton]")


def test_same_process_triton_then_tilelang(npu):
    """同进程: 先执行 triton matmul, 再执行 tilelang GQA, 双方结果均需正确."""
    run_triton_matmul_check(tag="[triton-then-tilelang]")
    _run_gqa_check(tag="[triton-then-tilelang]")


def test_same_process_interleaved(npu):
    """同进程交替执行: 每轮 GQA -> matmul, 连续多轮, 捕捉框架切换时的状态串扰."""
    for i in range(3):
        _run_gqa_check(tag=f"[interleave-r{i}]")
        run_triton_matmul_check(tag=f"[interleave-r{i}]")


# ---------------------------------------------------------------------------
# 场景 B: 跨进程
# ---------------------------------------------------------------------------

def test_cross_process_sequential(npu):
    """跨进程顺序执行: 两个 worker 先后运行, 双向顺序 (tilelang->triton / triton->tilelang) 各一遍."""
    for order, names in (
        ("tilelang->triton", WORKERS),
        ("triton->tilelang", tuple(reversed(WORKERS))),
    ):
        for name in names:
            path = os.path.join(_THIS_DIR, name)
            proc = subprocess.run(
                [sys.executable, path],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT,
            )
            assert proc.returncode == 0 and WORKER_PASS_MARK in proc.stdout, (
                _worker_fail_message(name, proc.returncode, proc.stdout, proc.stderr)
            )
            pass_lines = [ln for ln in proc.stdout.splitlines() if WORKER_PASS_MARK in ln]
            print(f"[cross-process:{order}] {name}: {pass_lines[-1] if pass_lines else 'PASS'}")


def test_cross_process_concurrent(npu):
    """跨进程并发执行: 两个 worker 同时启动, 在同一张 NPU 卡上并行, 验证资源不冲突."""
    procs = {}
    for name in WORKERS:
        procs[name] = subprocess.Popen(
            [sys.executable, os.path.join(_THIS_DIR, name)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    for name, proc in procs.items():
        try:
            out, err = proc.communicate(timeout=WORKER_TIMEOUT)
        except subprocess.TimeoutExpired:
            for p in procs.values():
                p.kill()
            pytest.fail(f"并发场景 worker {name} 超时 (> {WORKER_TIMEOUT}s), 已终止全部 worker")
        assert proc.returncode == 0 and WORKER_PASS_MARK in out, (
            _worker_fail_message(name, proc.returncode, out, err)
        )
        pass_lines = [ln for ln in out.splitlines() if WORKER_PASS_MARK in ln]
        print(f"[cross-process:concurrent] {name}: {pass_lines[-1] if pass_lines else 'PASS'}")
