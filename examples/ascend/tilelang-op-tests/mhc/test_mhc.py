# -*- coding: utf-8 -*-
"""mHC (shifted manifold-constrained hyper-connections) 算子 pytest 测试脚本 (自包含, 不依赖原测试工程).

来源参考 (yangsichan/tilelang-asc, commit 053ddb4):
    - kernel 实现:   examples/ascend/mhc/ (多文件: mhc_pre.py / mhc_pre_mk.py /
                     mhc_pre_large.py / mhc_post.py, 已逐字节复制到本目录)
    - 正确性用例:    examples/ascend/mhc/test_mhc_pre.py / test_mhc_post.py
    - 性能口径:      examples/ascend/mhc/README.md (bench_mhc_pre / bench_mhc_post)
    仓库地址: https://github.com/yangsichan/tilelang-asc/tree/053ddb470aefcbc6972d994739983734eeb4eaf8/examples/ascend/mhc

说明
    mHC Pre: 混合 Cube/Vector 一次 launch 完成投影 / RMS 统计 / 加权层输入,
    按 N 在 MK (K 切分) 与 Whole-M 两种实现间自动分发;
    mHC Post: 纯 Vector kernel, 用 Pre 输出的系数做残差流混合。
    本目录保留 kernel 的多文件结构 (与上游一致), 唯一改动是
    mhc_pre.py / mhc_pre_large.py 的包内相对导入改为平铺导入
    (本脚本会把 mhc/ 目录加入 sys.path); kernel 实现本身逐字保留,
    同步上游更新时整文件替换即可。

算子契约 (kernel 固定)
    Pre 输入:  x [N,4,H] BF16, prev_pre [N,4]/[N,4,1] FP32, phi [24,4H] FP32,
               alpha [3] FP32, bias [24] FP32 (全部连续且在同一 NPU);
    Pre 输出:  layer_input [N,H] BF16, next_pre_mix [N,4,1] FP32,
               post_mix [N,4,1] FP32, comb_mix [N,4,4] FP32;
    Post 输入: x [...,H] BF16, residual [...,4,H] BF16,
               post_layer_mix [...,4(,1)] FP32, comb_res_mix [...,4,4] FP32;
    Post 输出: [...,4,H] BF16 (组合矩阵按 [输入流, 输出流] 索引);
    H ∈ {4096, 5120, 7168}; N 取值 1..8192 (上游覆盖的 serving 范围)。

运行环境
    - Ascend NPU 服务器: CANN + torch_npu + tilelang (tilelang.ascend)
    - 容量默认 28 AIC / 56 AIV (与实际测试卡一致)
    - Pre 首次调用会额外 JIT 编译一个小型 C++ 分配扩展
      (需要 Ninja 与 C++ 编译器, 与上游要求一致)

运行方式
    pytest mhc/test_mhc.py -v -s                # 全部用例
    pytest mhc/test_mhc.py -v -k "routing"      # CPU 路由测试 (无需 NPU)
    pytest mhc/test_mhc.py -v -k "pre and not perf"
    pytest mhc/test_mhc.py -v -k "not perf"     # 跳过性能用例

精度规格 (fork 开发口径, 按输出类型区分)
    layer_input (BF16):        atol=0.03125, rtol=1e-3
    FP32 输出 (pre/post/comb): atol=5e-5,    rtol=1e-3
    Post 输出 (BF16):          atol=0.03125, rtol=1e-3

环境变量
    MHC_BENCH_BACKEND    性能计时 backend, 默认 msprof_detail (fork bench 口径)

shape 泛化用例表 (MHC_PRE_CASES / MHC_POST_CASES) 当前为 fork 测试全量移植
    (覆盖 MK/large 分发边界、零输入、残块尾、系数批次、多任务、批处理 Post、
    两种 post_mix 形态、非默认核数等); 具体泛化规格确认后直接替换/扩充该表即可。
"""

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest
import torch

# mhc_pre 顶层仅依赖标准库 (kernel 懒加载), CPU 路由测试可在无 NPU 环境运行
from mhc_pre import _select_kernel, bind_mhc_pre, mhc_pre

try:
    from mhc_post import mhc_post  # 顶层 import tilelang
except ImportError as exc:  # pragma: no cover - 仅在依赖缺失的环境触发
    pytest.skip(f"缺少运行依赖 (tilelang / mhc_post): {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. 输入生成与参考实现 (复制自 fork test_mhc_pre.py / test_mhc_post.py)
# ---------------------------------------------------------------------------

def _pre_inputs(n, hidden, device, *, zero=False, seed_offset=0):
    """fork test_mhc_pre._inputs: CPU 生成后搬到 device, 数值有界."""
    generator = torch.Generator(device="cpu").manual_seed(20261112 + n + hidden + seed_offset)
    x = torch.randn((n, 4, hidden), dtype=torch.bfloat16, generator=generator)
    if zero:
        x.zero_()
    previous = torch.randn((n, 4, 1), dtype=torch.float32, generator=generator).sigmoid().add_(1e-6)
    phi = torch.randn((24, 4 * hidden), dtype=torch.float32, generator=generator).div_(math.sqrt(4 * hidden))
    alpha = torch.randn((3,), dtype=torch.float32, generator=generator).mul_(0.1).add_(1)
    bias = torch.randn((24,), dtype=torch.float32, generator=generator).mul_(0.1)
    return tuple(tensor.to(device=device) for tensor in (x, previous, phi, alpha, bias))


def _pre_reference(x, previous, phi, alpha, bias):
    """fork test_mhc_pre._reference: 独立 CPU FP32 方程 (128 行分块控制中间量)."""
    n, _, hidden = x.shape
    phi, alpha, bias = (tensor.detach().float().cpu() for tensor in (phi, alpha, bias))
    previous = previous.detach().float().cpu().reshape(n, 4, 1)
    layer = torch.empty((n, hidden), dtype=torch.bfloat16)
    pre = torch.empty((n, 4, 1), dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((n, 4, 4), dtype=torch.float32)
    for start in range(0, n, 128):
        end = min(start + 128, n)
        residual = x[start:end].detach().float().cpu()
        flat = residual.reshape(end - start, 4 * hidden)
        mix = (flat @ phi.T) * (flat.square().mean(-1, keepdim=True) + 1e-20).rsqrt()
        pre[start:end, :, 0] = (mix[:, :4] * alpha[0] + bias[:4]).sigmoid() + 1e-6
        post[start:end, :, 0] = 2 * (mix[:, 4:8] * alpha[1] + bias[4:8]).sigmoid()
        matrix = (mix[:, 8:24] * alpha[2] + bias[8:24]).reshape(-1, 4, 4).softmax(-1) + 1e-6
        matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            matrix = matrix / (matrix.sum(-1, keepdim=True) + 1e-6)
            matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
        comb[start:end] = matrix
        layer[start:end] = (residual * previous[start:end]).sum(1).to(torch.bfloat16)
    return layer, pre, post, comb


def _assert_pre_outputs(actual, expected):
    """fork test_mhc_pre._assert_outputs: 按输出类型区分容差 (BF16 / FP32)."""
    assert isinstance(actual, (tuple, list)) and len(actual) == 4
    for index, (value, reference) in enumerate(zip(actual, expected)):
        assert value.shape == reference.shape
        assert value.dtype == reference.dtype
        assert value.device.type == "npu"
        actual_cpu = value.cpu()
        assert torch.isfinite(actual_cpu).all().item()
        torch.testing.assert_close(actual_cpu, reference, rtol=1e-3, atol=0.03125 if index == 0 else 5e-5)


def _post_inputs(batch, hidden, device):
    """fork test_mhc_post._inputs: 组合矩阵刻意非对称 (转置方向错误必须能测出)."""
    generator = torch.Generator(device="cpu").manual_seed(20261112 + hidden)
    x = torch.randn((*batch, hidden), dtype=torch.bfloat16, generator=generator)
    residual = torch.randn((*batch, 4, hidden), dtype=torch.bfloat16, generator=generator)
    post = torch.randn((*batch, 4, 1), dtype=torch.float32, generator=generator)
    comb = torch.randn((*batch, 4, 4), dtype=torch.float32, generator=generator)
    return tuple(value.to(device) for value in (x, residual, post, comb))


def _post_reference(x, residual, post, comb):
    """fork test_mhc_post._reference: CPU FP32 矩阵乘参考."""
    x, residual, post, comb = (value.detach().float().cpu() for value in (x, residual, post, comb))
    post = post.reshape(*residual.shape[:-1], 1)
    return (comb.transpose(-1, -2) @ residual + post * x.unsqueeze(-2)).bfloat16()


def _assert_post_output(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "npu"
    actual = actual.cpu()
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. shape 泛化用例表
#    TODO(shape 泛化规格): 当前为 fork 测试用例全量移植, 规格确认后替换/扩充。
#    字段说明:
#        hidden           4096 / 5120 / 7168 (kernel 固定三选一)
#        n                token 数 (serving 范围 1..8192)
#        zero             x 全零 (RMS 下界回归), 默认 False
#        prev_flat        prev_pre 用 [N,4] 形态 (默认 [N,4,1]), 默认 False
#        batch            Post 的前导 batch 维 (元组)
#        squeeze_mix      post_layer_mix 用 [...,4] 形态 (默认 [...,4,1])
#        cores            Post 的 num_vec_cores, 默认 56
#        perf             是否纳入性能用例, 默认 False
# ---------------------------------------------------------------------------

SERVING_N_MAX = 8192  # 上游 README: serving 覆盖范围 1..8192
PERF_WARMUP = 30      # fork bench 口径: 30 次预热
PERF_REPEAT = 500     # fork bench 口径: 500 次测量

MHC_PRE_CASES = [
    dict(hidden=4096, n=16),                     # mk 冒烟
    dict(hidden=4096, n=1, zero=True),           # mk-zero-rms
    dict(hidden=5120, n=33),                     # mk-row-tail
    dict(hidden=7168, n=129),                    # mk-coefficient-batches
    dict(hidden=4096, n=2305),                   # large-empty-aiv-half
    dict(hidden=5120, n=2400),                   # mk-m240-complete
    dict(hidden=7168, n=2401),                   # large-after-mk-complete
    dict(hidden=4096, n=3137, prev_flat=True),   # large-aiv-boundary-tail (prev 用 [N,4])
    dict(hidden=5120, n=3584),                   # mk-m256-complete
    dict(hidden=7168, n=3585),                   # large-natural-boundary
    dict(hidden=4096, n=8192, perf=True),        # large-multiple-jobs (README 基准点)
]

MHC_POST_CASES = [
    dict(batch=(1,), hidden=4096),                        # single-token
    dict(batch=(57,), hidden=4096, squeeze_mix=True),     # persistent-token-tail
    dict(batch=(85,), hidden=5120),                       # two-hidden-tiles-three-stages
    dict(batch=(2, 17), hidden=7168),                     # batched-hidden-tiles
    dict(batch=(17,), hidden=5120, squeeze_mix=True, cores=4),  # smaller-core-capacity
    dict(batch=(4096,), hidden=4096, perf=True),          # 大 N 性能参考
]


def _pre_case_id(cfg: dict) -> str:
    parts = [f"h{cfg['hidden']}", f"n{cfg['n']}"]
    if cfg.get("zero"):
        parts.append("zero")
    if cfg.get("prev_flat"):
        parts.append("flat")
    return ("pre_" + "_".join(parts)).lower()


def _post_case_id(cfg: dict) -> str:
    batch = "x".join(str(b) for b in cfg["batch"])
    parts = [f"h{cfg['hidden']}", f"b{batch}"]
    if cfg.get("squeeze_mix"):
        parts.append("sq")
    if cfg.get("cores", 56) != 56:
        parts.append(f"c{cfg['cores']}")
    return ("post_" + "_".join(parts)).lower()


def validate_pre_case(cfg: dict) -> None:
    tag = _pre_case_id(cfg)
    if cfg["hidden"] not in (4096, 5120, 7168):
        raise ValueError(f"[{tag}] hidden 仅支持 4096/5120/7168, 当前 {cfg['hidden']}")
    if not 0 < cfg["n"] <= SERVING_N_MAX:
        raise ValueError(f"[{tag}] n={cfg['n']} 超出 serving 范围 1..{SERVING_N_MAX}")
    _select_kernel(cfg["n"], cfg["hidden"])  # kernel 自带几何校验


def validate_post_case(cfg: dict) -> None:
    tag = _post_case_id(cfg)
    if cfg["hidden"] not in (4096, 5120, 7168):
        raise ValueError(f"[{tag}] hidden 仅支持 4096/5120/7168, 当前 {cfg['hidden']}")
    tokens = math.prod(cfg["batch"])
    if not 0 < tokens <= SERVING_N_MAX:
        raise ValueError(f"[{tag}] token 数 {tokens} 超出 serving 范围 1..{SERVING_N_MAX}")
    if not isinstance(cfg.get("cores", 56), int) or cfg.get("cores", 56) <= 0:
        raise ValueError(f"[{tag}] cores 必须为正整数")


PRE_CASE_IDS = [_pre_case_id(c) for c in MHC_PRE_CASES]
PRE_PERF_CASES = [c for c in MHC_PRE_CASES if c.get("perf")]
PRE_PERF_IDS = [_pre_case_id(c) for c in PRE_PERF_CASES]
POST_CASE_IDS = [_post_case_id(c) for c in MHC_POST_CASES]
POST_PERF_CASES = [c for c in MHC_POST_CASES if c.get("perf")]
POST_PERF_IDS = [_post_case_id(c) for c in POST_PERF_CASES]


# ---------------------------------------------------------------------------
# 3. CPU 路由测试 (无需 NPU, fork test_mhc_pre 同款)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,hidden,route",
    [
        (2304, 4096, "mk"),
        (2305, 4096, "large"),
        (2400, 4096, "mk"),
        (2401, 4096, "large"),
        (2560, 5120, "mk"),
        (2561, 5120, "large"),
        (3136, 7168, "large"),
        (3584, 7168, "mk"),
        (3585, 7168, "large"),
        (8192, 4096, "large"),
    ],
)
def test_routing_preserves_original_mk_complete_tiles(n, hidden, route):
    assert _select_kernel(n, hidden)["route"] == route


@pytest.mark.parametrize(
    "n,aic,aiv,route",
    [
        (2305, 24, 48, "mk"),
        (3073, 24, 48, "large"),
        (2305, 32, 56, "mk"),
        (3585, 32, 56, "large"),
        (3585, 32, 64, "mk"),
        (4097, 32, 64, "large"),
    ],
)
def test_other_core_capacities_retain_their_partition(n, aic, aiv, route):
    assert _select_kernel(n, 4096, num_aic_cores=aic, num_vec_cores=aiv)["route"] == route


@pytest.mark.parametrize(
    "n,hidden,aic,aiv",
    [
        (0, 4096, 28, 56),
        (-1, 4096, 28, 56),
        (True, 4096, 28, 56),
        (16, 4097, 28, 56),
        (1, 4096, 0, 56),
        (1, 4096, 28, 1),
    ],
)
def test_invalid_geometry_is_rejected(n, hidden, aic, aiv):
    with pytest.raises(ValueError):
        _select_kernel(n, hidden, num_aic_cores=aic, num_vec_cores=aiv)


def test_case_configs_satisfy_kernel_constraints():
    """静态校验两张用例表满足 kernel 约束 (无需 NPU, 秒级)."""
    assert MHC_PRE_CASES and MHC_POST_CASES
    for cfg in MHC_PRE_CASES:
        validate_pre_case(cfg)
    for cfg in MHC_POST_CASES:
        validate_post_case(cfg)


# ---------------------------------------------------------------------------
# 4. 设备侧正确性用例
# ---------------------------------------------------------------------------

@pytest.fixture
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    return torch.device("npu", torch.npu.current_device())


def test_pre_cpu_inputs_are_rejected_before_device_compilation():
    with pytest.raises(ValueError, match="NPU|npu"):
        bind_mhc_pre(*_pre_inputs(1, 4096, "cpu"))


def test_post_cpu_inputs_are_rejected_before_device_compilation():
    with pytest.raises(ValueError, match="NPU|npu"):
        mhc_post(*_post_inputs((1,), 4096, "cpu"))


@pytest.mark.parametrize(
    "invalid",
    ["empty", "hidden", "mix_shape", "cores_zero", "cores_bool"],
)
def test_post_invalid_geometry_is_rejected(invalid):
    inputs = list(_post_inputs((0,) if invalid == "empty" else (1,), 128 if invalid == "hidden" else 4096, "cpu"))
    if invalid == "mix_shape":
        inputs[2] = torch.empty((1, 5), dtype=torch.float32)
    cores = 0 if invalid == "cores_zero" else True if invalid == "cores_bool" else 56
    with pytest.raises(ValueError, match="token count|post_layer_mix|num_vec_cores"):
        mhc_post(*inputs, num_vec_cores=cores)


@pytest.mark.parametrize("cfg", MHC_PRE_CASES, ids=PRE_CASE_IDS)
def test_mhc_pre_correctness(cfg, npu_device):
    """Pre 正确性: 四路输出对 CPU FP32 参考方程 (fork 判据)."""
    inputs = list(_pre_inputs(cfg["n"], cfg["hidden"], npu_device, zero=cfg.get("zero", False)))
    if cfg.get("prev_flat"):
        inputs[1] = inputs[1].reshape(cfg["n"], 4)
    expected = _pre_reference(*inputs)
    with torch.inference_mode():
        actual = mhc_pre(*inputs)
    torch.npu.synchronize(npu_device)
    _assert_pre_outputs(actual, expected)
    route = _select_kernel(cfg["n"], cfg["hidden"])["route"]
    print(f"[correctness] {_pre_case_id(cfg)} route={route} PASS")


@pytest.mark.parametrize("cfg", MHC_POST_CASES, ids=POST_CASE_IDS)
def test_mhc_post_correctness(cfg, npu_device):
    """Post 正确性: 非对称组合矩阵 + CPU 矩阵乘参考 (方向错误会被测出)."""
    inputs = list(_post_inputs(cfg["batch"], cfg["hidden"], npu_device))
    if cfg.get("squeeze_mix"):
        inputs[2] = inputs[2].squeeze(-1)
    expected = _post_reference(*inputs)
    with torch.inference_mode():
        actual = mhc_post(*inputs, num_vec_cores=cfg.get("cores", 56))
    torch.npu.synchronize(npu_device)
    _assert_post_output(actual, expected)
    print(f"[correctness] {_post_case_id(cfg)} PASS")


@pytest.mark.parametrize("n,hidden", [(33, 5120), (2305, 4096)], ids=["mk", "large"])
def test_pre_bound_launch_uses_current_stream_and_fresh_outputs(npu_device, n, hidden):
    """bind_mhc_pre: 换流执行 + 原地更新参数 + 每次全新输出 (fork 同名回归)."""
    inputs = _pre_inputs(n, hidden, npu_device)
    run = bind_mhc_pre(*inputs)
    with torch.inference_mode():
        first = run()
        torch.npu.synchronize(npu_device)
        _assert_pre_outputs(first, _pre_reference(*inputs))
        saved_first = tuple(tensor.cpu().clone() for tensor in first)

        stream = torch.npu.Stream(device=npu_device)
        with torch.npu.stream(stream):
            inputs[1].add_(0.125)
            inputs[2].mul_(-0.75)
            inputs[4].add_(0.2)
            second = run()
        stream.synchronize()
        _assert_pre_outputs(second, _pre_reference(*inputs))

    assert all(before.data_ptr() != after.data_ptr() for before, after in zip(first, second))
    for before, saved in zip(first, saved_first):
        torch.testing.assert_close(before.cpu(), saved, rtol=0, atol=0)


def test_pre_functional_cache_accepts_new_input_objects(npu_device):
    """同一 shape 的新输入对象命中缓存, 且各自产出新输出 (fork 同名回归)."""
    first_inputs = _pre_inputs(33, 5120, npu_device)
    second_inputs = _pre_inputs(33, 5120, npu_device, seed_offset=1)
    with torch.inference_mode():
        first = mhc_pre(*first_inputs)
        second = mhc_pre(*second_inputs)
    torch.npu.synchronize(npu_device)
    _assert_pre_outputs(first, _pre_reference(*first_inputs))
    _assert_pre_outputs(second, _pre_reference(*second_inputs))
    assert all(before.data_ptr() != after.data_ptr() for before, after in zip(first, second))


def test_post_current_stream_fresh_outputs_and_new_inputs(npu_device):
    """Post: 换流执行 + 同 shape 新输入对象 + 全新输出 (fork 同名回归)."""
    inputs = _post_inputs((29,), 5120, npu_device)
    with torch.inference_mode():
        first = mhc_post(*inputs)
        torch.npu.synchronize(npu_device)
        _assert_post_output(first, _post_reference(*inputs))
        saved = first.cpu().clone()
        stream = torch.npu.Stream(device=npu_device)
        with torch.npu.stream(stream):
            second_inputs = (inputs[0].neg(), inputs[1], inputs[2] * 0.75, inputs[3])
            second = mhc_post(*second_inputs)
        stream.synchronize()
        _assert_post_output(second, _post_reference(*second_inputs))
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), saved, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["dtype", "noncontiguous"])
def test_pre_invalid_npu_inputs_are_rejected(npu_device, invalid):
    inputs = list(_pre_inputs(1, 4096, npu_device))
    if invalid == "dtype":
        inputs[0] = inputs[0].float()
    else:
        inputs[2] = torch.empty((24, 8 * 4096), dtype=torch.float32, device=npu_device)[:, ::2]
    with pytest.raises(ValueError):
        bind_mhc_pre(*inputs)


@pytest.mark.parametrize("invalid", ["dtype", "noncontiguous", "batch_shape", "device"])
def test_post_invalid_device_inputs_are_rejected(npu_device, invalid):
    inputs = list(_post_inputs((2,), 4096, npu_device))
    if invalid == "dtype":
        inputs[0] = inputs[0].float()
    elif invalid == "noncontiguous":
        inputs[3] = inputs[3].transpose(-1, -2)
    elif invalid == "batch_shape":
        inputs[0] = inputs[0].view(1, 2, 4096)
    else:
        inputs[3] = inputs[3].cpu()
    with pytest.raises(ValueError):
        mhc_post(*inputs)


def test_pre_outputs_feed_post_on_the_same_stream(npu_device):
    """Pre/Post 集成: 系数不经 host 同步直接喂给 Post (fork 同名回归)."""
    n, hidden = 33, 4096
    x, residual, previous, _ = _post_inputs((n,), hidden, npu_device)
    phi = torch.randn((24, 4 * hidden), device=npu_device, dtype=torch.float32) / (4 * hidden) ** 0.5
    alpha = torch.ones(3, device=npu_device, dtype=torch.float32)
    bias = torch.zeros(24, device=npu_device, dtype=torch.float32)
    with torch.inference_mode():
        layer_input, _, post, comb = mhc_pre(residual, previous, phi, alpha, bias)
        layer_output = (layer_input + x).contiguous()
        actual = mhc_post(layer_output, residual, post, comb)
    torch.npu.synchronize(npu_device)
    _assert_post_output(actual, _post_reference(layer_output, residual, post, comb))
    print("[correctness] pre_post_integration PASS")


# ---------------------------------------------------------------------------
# 5. 性能用例 (fork bench 口径: msprof_detail, 30 预热 / 500 测量)
#    计时 lambda 含每次调用的新鲜输出分配 (Pre 还含 MK workspace 分配),
#    编译/绑定/首次同步执行均在计时外, 与 fork bench README 一致。
# ---------------------------------------------------------------------------

def _bench_ms(fn, backend, warmup, repeat):
    """单次 do_bench 计时, 返回毫秒 (兼容 msprof 统计对象与 float 两种返回)."""
    from tilelang.profiler import do_bench

    prof = do_bench(fn, backend=backend, _n_warmup=warmup, _n_repeat=repeat)
    return (prof.dur_ns / 1e6) if hasattr(prof, "dur_ns") else float(prof)


@pytest.mark.parametrize("cfg", PRE_PERF_CASES, ids=PRE_PERF_IDS)
def test_mhc_pre_perf(cfg, npu_device):
    """Pre 性能: bind 后计时, tilelang 耗时以 ms (3 位小数) 输出 (fork bench 口径)."""
    inputs = _pre_inputs(cfg["n"], cfg["hidden"], npu_device)
    run = bind_mhc_pre(*inputs)  # 绑定输入, 计时只含执行 + 新鲜输出分配
    with torch.inference_mode():
        run()
    torch.npu.synchronize(npu_device)

    backend = os.environ.get("MHC_BENCH_BACKEND", "msprof_detail")
    ms = _bench_ms(lambda: run(), backend, PERF_WARMUP, PERF_REPEAT)
    route = _select_kernel(cfg["n"], cfg["hidden"])["route"]
    print(f"[perf][tilelang] {_pre_case_id(cfg)}: {ms:.3f} ms | pre route={route} | {backend} ({PERF_WARMUP}w/{PERF_REPEAT}r)")
    assert ms > 0


@pytest.mark.parametrize("cfg", POST_PERF_CASES, ids=POST_PERF_IDS)
def test_mhc_post_perf(cfg, npu_device):
    """Post 性能: 计时含验证/输出分配, tilelang 耗时以 ms (3 位小数) 输出 (fork bench 口径)."""
    inputs = _post_inputs(cfg["batch"], cfg["hidden"], npu_device)
    with torch.inference_mode():
        mhc_post(*inputs, num_vec_cores=cfg.get("cores", 56))
    torch.npu.synchronize(npu_device)

    backend = os.environ.get("MHC_BENCH_BACKEND", "msprof_detail")
    ms = _bench_ms(
        lambda: mhc_post(*inputs, num_vec_cores=cfg.get("cores", 56)),
        backend, PERF_WARMUP, PERF_REPEAT,
    )
    print(f"[perf][tilelang] {_post_case_id(cfg)}: {ms:.3f} ms | post | {backend} ({PERF_WARMUP}w/{PERF_REPEAT}r)")
    assert ms > 0
