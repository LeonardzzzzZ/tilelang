"""T.gemm 泛化测试（pytest，独立单算子脚本）。

被测算子：``T.gemm``（TileLang Ascend 语言层），规格来源
``docs/requirements/WIKI2026112514203848 T.gemm.md``。

设备契约（为什么输入必须在 NPU 上）
------------------------------------
``tilelang.compile(..., target="ascend", out_idx=...)`` 默认走 tvm_ffi 执行后端的
"callee-allocated output" ABI：调用时会把**第一个 torch.Tensor 输入**作为
allocator anchor 追加传给生成的 host stub，anchor 的 DLPack 设备类型必须等于
编译目标（ascend target 注册为 ``kDLExtDev = 12``，与 torch_npu 的 NPU 张量
一致），输出 C 也在 anchor 所在设备上分配。传入 CPU 张量（``kDLCPU = 1``）
时断言失败，报：

    RuntimeError: kernel: allocator anchor has the wrong device type;
    expected: 1, got: 12

（消息里 expected/got 的打印顺序与直觉相反：1 是 anchor 的实际类型 CPU，
12 是 kernel 编译目标 NPU。）因此本脚本：

- 导入 ``torch_npu`` 注册 ``npu`` 设备，无 NPU 的机器上用例自动 skip；
- 输入在 CPU 上按固定种子生成（可复现，且 int8/fp8 均可生成），再搬运到
  NPU 喂给 kernel；
- 参考结果在 CPU 上以 float64 计算（Ascend 不支持 fp64 matmul）。

覆盖维度
--------
- 入口：L1（仅 NT）与 L0（NN / NT / TN / TT 四种转置组合）。
- 数据类型：float16 / bfloat16 / float32 / int8 / float8_e4m3fn。
- 形状：对齐形状、非对齐形状（L1 补零后裁剪、L0 直接按有效区域计算）、
  最小形状、较大形状。
- 输出类型：float32 / float16 / bfloat16（浮点），int32（int8 输入）。
- 累加：``clear_accum=True`` 与 ``clear_accum=False``（连续两段累加）。

判定基准
--------
参考实现用 PyTorch 在 float64 下计算 ``op(A) @ op(B)``，再按输出类型舍入；
被测 kernel 的结果与之逐元素比较。容差按输出类型区分（见 ``_TOL``）。

不覆盖（规格明确排除或需外层分块）
----------------------------------
- L1 入口的 NN / TN / TT 组合（规格标注“不支持直接调用”）。
- 单行矩阵（规格标注“未来支持”）、零尺寸（不在规格范围内）。
- A、B 使用不同类型（不支持）、fp8_e5m2（当前未接入）、fp4_e2m1fn（不支持）。
- 超出片上缓存容量的整张矩阵（需外层分块循环，单次 ``T.gemm`` 不覆盖）。
- ``policy`` 仅支持 ``GemmWarpPolicy.Square``（默认值，不单独泛化）。
- ``unit_flag_ctrl`` / ``annotations`` 需与搬出配套，暂不纳入泛化矩阵。

运行方式（独立执行，不进入仓库回归清单 testcases.txt）
--------
    python -m pytest standalone/gemm/test_gemm_general.py -v
    # 或整文件拷回 tilelang 仓库后:
    python -m pytest testcases/functional/test_gemm_general.py -v

脚本假定运行环境已就绪（tilelang + torch + pytest，Ascend 后端可用）；
无 NPU 时用例自动 skip。
"""

import hashlib

import pytest
import torch

import tilelang
from tilelang.ascend import language as T


# ---------------------------------------------------------------------------
# NPU 设备检测
# ---------------------------------------------------------------------------

def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  # 导入即注册 npu 设备
    except Exception:
        pass
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture(scope="module")
def npu_device():
    if not _npu_available():
        pytest.skip("未检测到可用 NPU (需要 torch_npu + CANN 环境)")
    return torch.device("npu")


# ---------------------------------------------------------------------------
# 规格常量
# ---------------------------------------------------------------------------

# 输入位宽对应的 C0（L1 计算 K 的对齐单位）：C0 = 256 / bitwidth。
_C0 = {
    "float32": 8,
    "float16": 16,
    "bfloat16": 16,
    "int8": 32,
    "float8_e4m3fn": 32,
}

# 累加类型：int8 累加到 int32，其余累加到 float32。
_ACC_DTYPE = {
    "float32": "float32",
    "float16": "float32",
    "bfloat16": "float32",
    "int8": "int32",
    "float8_e4m3fn": "float32",
}

# 输出比较容差 (rtol, atol)，依据输出类型。
_TOL = {
    "float32": (1e-3, 1e-3),
    "float16": (1.6e-2, 1.6e-2),
    "bfloat16": (1.6e-2, 1.6e-2),
    "int32": (0.0, 0.0),
}

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


# ---------------------------------------------------------------------------
# 用例定义
# ---------------------------------------------------------------------------

class GemmCase:
    """一个 T.gemm 泛化用例。

    参数
    ----
    entry        : ``"l1"`` 或 ``"l0"``，输入所在层级。
    M, N, K      : 逻辑矩阵维度，结果形状为 ``[M, N]``。
    dtype        : A、B 的 TileLang 类型名。
    transpose_a  : 是否转置 A。
    transpose_b  : 是否转置 B。
    out_dtype    : 输出（GM）类型。
    accumulate   : True 时连续两段 GEMM，第二段 ``clear_accum=False``，
                   期望结果为 2 * op(A) @ op(B)。
    note         : 备注。
    """

    __slots__ = ("id", "entry", "M", "N", "K", "dtype",
                 "transpose_a", "transpose_b", "out_dtype", "accumulate", "note")

    def __init__(self, id, entry, M, N, K, dtype="float16",
                 transpose_a=False, transpose_b=True,
                 out_dtype="float32", accumulate=False, note=""):
        self.id = id
        self.entry = entry
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.out_dtype = out_dtype
        self.accumulate = accumulate
        self.note = note


CASES = [
    # ---- L1 入口：仅 NT（transpose_A=False, transpose_B=True）----
    GemmCase("l1_nt_fp16_64x64x64", "l1", 64, 64, 64),
    GemmCase("l1_nt_fp16_128x128x512", "l1", 128, 128, 512),
    GemmCase("l1_nt_fp16_16x16x16", "l1", 16, 16, 16, note="最小对齐形状"),
    GemmCase("l1_nt_fp16_17x19x24_pad", "l1", 17, 19, 24,
             note="非对齐有效尺寸，L1 补零后裁剪输出"),
    GemmCase("l1_nt_bf16_128x128x256", "l1", 128, 128, 256, dtype="bfloat16"),
    GemmCase("l1_nt_fp32_64x64x64", "l1", 64, 64, 64, dtype="float32"),
    GemmCase("l1_nt_fp32_17x24x24_pad", "l1", 17, 24, 24, dtype="float32",
             note="规格示例形状"),
    GemmCase("l1_nt_fp32_128x128x256", "l1", 128, 128, 256, dtype="float32"),
    GemmCase("l1_nt_int8_64x64x64", "l1", 64, 64, 64, dtype="int8",
             out_dtype="int32"),
    GemmCase("l1_nt_int8_128x128x256", "l1", 128, 128, 256, dtype="int8",
             out_dtype="int32"),
    GemmCase("l1_nt_fp8e4m3_64x64x64", "l1", 64, 64, 64, dtype="float8_e4m3fn"),
    GemmCase("l1_nt_fp16_128x128x256_out_fp16", "l1", 128, 128, 256,
             out_dtype="float16", note="L0C fp32 随路转 fp16"),
    GemmCase("l1_nt_fp16_64x64x64_out_bf16", "l1", 64, 64, 64,
             out_dtype="bfloat16", note="L0C fp32 随路转 bf16"),
    GemmCase("l1_nt_bf16_64x64x64_out_fp16", "l1", 64, 64, 64,
             dtype="bfloat16", out_dtype="float16"),
    GemmCase("l1_nt_fp16_64x64x64_accum", "l1", 64, 64, 64, accumulate=True,
             note="clear_accum=False 累加"),
    GemmCase("l1_nt_fp32_64x64x64_accum", "l1", 64, 64, 64, dtype="float32",
             accumulate=True),

    # ---- L0 入口：NN ----
    GemmCase("l0_nn_fp16_64x64x64", "l0", 64, 64, 64,
             transpose_a=False, transpose_b=False),
    GemmCase("l0_nn_fp16_17x19x24", "l0", 17, 19, 24,
             transpose_a=False, transpose_b=False, note="有效尺寸非对齐"),
    GemmCase("l0_nn_fp32_64x64x64", "l0", 64, 64, 64, dtype="float32",
             transpose_a=False, transpose_b=False),
    GemmCase("l0_nn_int8_64x64x64", "l0", 64, 64, 64, dtype="int8",
             transpose_a=False, transpose_b=False, out_dtype="int32"),
    GemmCase("l0_nn_bf16_128x128x64", "l0", 128, 128, 64, dtype="bfloat16",
             transpose_a=False, transpose_b=False),
    GemmCase("l0_nn_fp16_64x64x64_out_bf16", "l0", 64, 64, 64,
             transpose_a=False, transpose_b=False, out_dtype="bfloat16"),
    GemmCase("l0_nn_fp16_128x128x128", "l0", 128, 128, 128,
             transpose_a=False, transpose_b=False),

    # ---- L0 入口：NT ----
    GemmCase("l0_nt_fp16_64x64x64", "l0", 64, 64, 64),
    GemmCase("l0_nt_fp16_125x125x125", "l0", 125, 125, 125,
             note="规格尾块示例尺寸"),
    GemmCase("l0_nt_fp32_17x24x24", "l0", 17, 24, 24, dtype="float32"),
    GemmCase("l0_nt_fp8e4m3_64x64x64", "l0", 64, 64, 64,
             dtype="float8_e4m3fn"),
    GemmCase("l0_nt_fp16_64x64x64_accum", "l0", 64, 64, 64, accumulate=True),
    GemmCase("l0_nt_fp16_16x16x16", "l0", 16, 16, 16, note="最小对齐形状"),

    # ---- L0 入口：TN ----
    GemmCase("l0_tn_fp16_64x64x64", "l0", 64, 64, 64,
             transpose_a=True, transpose_b=False),
    GemmCase("l0_tn_fp16_33x47x29", "l0", 33, 47, 29,
             transpose_a=True, transpose_b=False, note="有效尺寸非对齐"),

    # ---- L0 入口：TT ----
    GemmCase("l0_tt_fp16_64x64x64", "l0", 64, 64, 64,
             transpose_a=True, transpose_b=True),
    GemmCase("l0_tt_bf16_100x70x50", "l0", 100, 70, 50, dtype="bfloat16",
             transpose_a=True, transpose_b=True),
    GemmCase("l0_tt_int8_128x96x64", "l0", 128, 96, 64, dtype="int8",
             transpose_a=True, transpose_b=True, out_dtype="int32"),
]


# ---------------------------------------------------------------------------
# 形状与规格校验
# ---------------------------------------------------------------------------

def _align(x, a):
    return ((x + a - 1) // a) * a


def _padded(case):
    """返回补零后的缓冲区维度 (Mp, Np, Kp)。

    M、N 对齐到 16；K 先按 C0 对齐，再对齐到 16，兼顾分形存储要求。
    """
    c0 = _C0[case.dtype]
    mp = _align(case.M, 16)
    np = _align(case.N, 16)
    kp = _align(_align(case.K, c0), 16)
    return mp, np, kp


def _validate(case):
    assert case.entry in ("l1", "l0"), case.id
    assert case.M > 0 and case.N > 0 and case.K > 0, case.id
    if case.entry == "l1":
        assert not case.transpose_a and case.transpose_b, \
            f"{case.id}: L1 入口仅支持 NT 组合"
        assert case.M <= 4080 and case.N <= 4080, \
            f"{case.id}: L1 的 M、N 不得超过 4080"
    else:
        assert case.M <= 4095 and case.N <= 4095 and case.K <= 4095, \
            f"{case.id}: L0 有效尺寸不得超过 4095"
    if case.dtype == "int8":
        assert case.out_dtype == "int32", f"{case.id}: int8 输入仅支持 int32 输出"
    else:
        assert case.out_dtype in ("float32", "float16", "bfloat16"), case.id


# ---------------------------------------------------------------------------
# kernel 构造
# ---------------------------------------------------------------------------
#
# 说明：``T.set_hf32_mode(None)`` 按规格示例关闭 HF32，这里对所有 dtype 统一
# 调用；HF32 仅影响 FP32 计算模式，对其它 dtype 视为无操作。

def _l1_nt(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((mp, kp), dt)
            b = T.alloc_l1((np, kp), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:mp, :kp], a)
            T.copy(B[:np, :kp], b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.copy(acc[:M, :N], C)

    return kernel


def _l1_nt_accum(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((mp, kp), dt)
            b = T.alloc_l1((np, kp), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:mp, :kp], a)
            T.copy(B[:np, :kp], b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=False)
            T.copy(acc[:M, :N], C)

    return kernel


def _l0_nn(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((K, N), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((mp, kp), dt)
            b = T.alloc_l1((kp, np), dt)
            a0 = T.alloc_l0a((mp, kp), dt)
            b0 = T.alloc_l0b((kp, np), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:mp, :kp], a)
            T.copy(B[:kp, :np], b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:M, :K], b0[:K, :N], acc[:M, :N], clear_accum=True)
            T.copy(acc[:M, :N], C)

    return kernel


def _l0_nt(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((mp, kp), dt)
            b = T.alloc_l1((np, kp), dt)
            a0 = T.alloc_l0a((mp, kp), dt)
            b0 = T.alloc_l0b((np, kp), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:mp, :kp], a)
            T.copy(B[:np, :kp], b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:M, :K], b0[:N, :K], acc[:M, :N],
                   transpose_B=True, clear_accum=True)
            T.copy(acc[:M, :N], C)

    return kernel


def _l0_nt_accum(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((mp, kp), dt)
            b = T.alloc_l1((np, kp), dt)
            a0 = T.alloc_l0a((mp, kp), dt)
            b0 = T.alloc_l0b((np, kp), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:mp, :kp], a)
            T.copy(B[:np, :kp], b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:M, :K], b0[:N, :K], acc[:M, :N],
                   transpose_B=True, clear_accum=True)
            T.gemm(a0[:M, :K], b0[:N, :K], acc[:M, :N],
                   transpose_B=True, clear_accum=False)
            T.copy(acc[:M, :N], C)

    return kernel


def _l0_tn(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((K, M), dt),
        B: T.Tensor((K, N), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((kp, mp), dt)
            b = T.alloc_l1((kp, np), dt)
            a0 = T.alloc_l0a((kp, mp), dt)
            b0 = T.alloc_l0b((kp, np), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:kp, :mp], a)
            T.copy(B[:kp, :np], b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:K, :M], b0[:K, :N], acc[:M, :N],
                   transpose_A=True, clear_accum=True)
            T.copy(acc[:M, :N], C)

    return kernel


def _l0_tt(case):
    M, N, K = case.M, case.N, case.K
    mp, np, kp = _padded(case)
    dt, acc_dt, out_dt = case.dtype, _ACC_DTYPE[case.dtype], case.out_dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((K, M), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), out_dt),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((kp, mp), dt)
            b = T.alloc_l1((np, kp), dt)
            a0 = T.alloc_l0a((kp, mp), dt)
            b0 = T.alloc_l0b((np, kp), dt)
            acc = T.alloc_l0c((mp, np), acc_dt)
            if dt == "float32":
                T.set_hf32_mode(None)
            T.copy(A[:kp, :mp], a)
            T.copy(B[:np, :kp], b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:K, :M], b0[:N, :K], acc[:M, :N],
                   transpose_A=True, transpose_B=True, clear_accum=True)
            T.copy(acc[:M, :N], C)

    return kernel


def _build_kernel(case):
    if case.entry == "l1":
        return _l1_nt_accum(case) if case.accumulate else _l1_nt(case)
    combo = (case.transpose_a, case.transpose_b)
    if combo == (False, False):
        return _l0_nn(case)
    if combo == (False, True):
        return _l0_nt_accum(case) if case.accumulate else _l0_nt(case)
    if combo == (True, False):
        return _l0_tn(case)
    return _l0_tt(case)


# ---------------------------------------------------------------------------
# 输入生成与参考实现
# ---------------------------------------------------------------------------

def _seed(case):
    return int(hashlib.md5(case.id.encode("utf-8")).hexdigest()[:8], 16)


def _make_inputs(case, device):
    """生成随机输入，返回 (A, B, ref)。

    输入在 CPU 上按固定种子生成（torch_npu 的设备侧 randn 不支持 CPU
    Generator，为保证可复现且 int8/fp8 均可生成，统一 CPU 生成）；
    计算完参考结果后再搬运到 ``device``（NPU）——ascend kernel 的 host stub
    要求输入张量位于目标设备，CPU 张量会触发 allocator anchor 断言失败。
    """
    a_shape = (case.K, case.M) if case.transpose_a else (case.M, case.K)
    b_shape = (case.N, case.K) if case.transpose_b else (case.K, case.N)
    g = torch.Generator(device="cpu").manual_seed(_seed(case))
    tdt = _TORCH_DTYPE[case.dtype]
    if case.dtype == "int8":
        # 限制取值范围，避免 int32 累加溢出。
        A = torch.randint(-8, 8, a_shape, generator=g, dtype=torch.int8)
        B = torch.randint(-8, 8, b_shape, generator=g, dtype=torch.int8)
    else:
        A = torch.randn(a_shape, generator=g).to(tdt)
        B = torch.randn(b_shape, generator=g).to(tdt)
    ref = _reference(case, A, B)
    A = A.to(device)
    B = B.to(device)
    return A, B, ref


def _reference(case, A, B):
    """float64 下的参考结果，避免参考实现自身引入误差。

    一律在 CPU 上计算：Ascend NPU 不支持 fp64 matmul。
    """
    a = A.detach().to("cpu").to(torch.float32).to(torch.float64)
    b = B.detach().to("cpu").to(torch.float32).to(torch.float64)
    if case.transpose_a:
        a = a.transpose(0, 1)
    if case.transpose_b:
        b = b.transpose(0, 1)
    ref = a @ b
    if case.accumulate:
        ref = ref * 2
    return ref


def _expected(case, ref):
    if case.out_dtype == "int32":
        return ref.round().to(torch.int32)
    return ref.to(_TORCH_DTYPE[case.out_dtype])


def _assert_close(case, actual, ref):
    expected = _expected(case, ref)
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.cpu()

    if case.out_dtype == "int32":
        assert torch.equal(actual_cpu, expected_cpu), (
            f"{case.id}: int32 输出不相等，"
            f"不匹配元素 {int((actual_cpu != expected_cpu).sum())}/{actual_cpu.numel()}"
        )
        return

    a = actual_cpu.to(torch.float64)
    e = expected_cpu.to(torch.float64)
    rtol, atol = _TOL[case.out_dtype]
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    bad = diff > tol
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{case.id}: {n_bad}/{a.numel()} 个元素超差 "
        f"(rtol={rtol}, atol={atol})，最大绝对误差 {float(diff.max()):.6e}"
    )


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_gemm_general(case, npu_device):
    _validate(case)
    kernel = _build_kernel(case)
    compiled = tilelang.compile(kernel, target="ascend", out_idx=2)
    A, B, ref = _make_inputs(case, npu_device)
    C = compiled(A, B)
    torch.npu.synchronize()
    _assert_close(case, C, ref)
