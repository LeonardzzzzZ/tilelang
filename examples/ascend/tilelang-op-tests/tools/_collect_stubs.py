# -*- coding: utf-8 -*-
"""pytest 收集期依赖 stub 插件 (仅用于 --collect-only 收集用例清单).

目的: 在没有 torch / torch_npu / tilelang / triton 的机器上 (如本地办公机)
也能完成 `pytest --collect-only`, 生成完整的用例节点 ID 清单 testcases.txt。

规则: 真实导入优先 —— 模块能正常导入时完全不干预; 只有导入失败才注册 stub。
因此在依赖齐全的服务器上运行本插件与不带插件结果一致 (stub 不会生效)。

用法 (在仓库根目录):
    python -m pytest --collect-only -q -p tools._collect_stubs

注意: stub 只保证 "收集" (参数化展开/用例发现) 可用, 不保证用例可执行;
执行测试仍需真实环境。generate_testcases.py 会自动携带本插件。
"""

import importlib
import sys
import types


def _passthrough_decorator(*args, **kwargs):
    def deco(fn):
        return fn

    return deco


def _ensure_module(name):
    """返回 (module, stubbed): 已导入/可导入则用真实模块, 否则注册并返回 stub."""
    if name in sys.modules:
        return sys.modules[name], False
    try:
        return importlib.import_module(name), False
    except Exception:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod, True


def _ensure_parent_attr(parent_name, child_name):
    """`from pkg import sub` 兜底: 在父包上补子模块属性。"""
    parent = sys.modules.get(parent_name)
    child = sys.modules.get(child_name)
    if parent is not None and child is not None and not hasattr(parent, child_name):
        setattr(parent, child_name, child)


def _set_default(mod, **attrs):
    for key, value in attrs.items():
        if not hasattr(mod, key):
            setattr(mod, key, value)


class _PassConfigKey:
    TL_ENABLE_FAST_MATH = "tl_enable_fast_math"


def install():
    # ---- torch ----
    torch, stubbed = _ensure_module("torch")
    if stubbed:
        _set_default(torch, bfloat16="bf16", float32="f32", float16="f16", Tensor="Tensor")
    _ensure_module("torch_npu")

    # ---- triton / triton-ascend ----
    triton, stubbed = _ensure_module("triton")
    if stubbed:
        _set_default(
            triton,
            Config=lambda d: ("config", d),
            autotune=_passthrough_decorator,
            jit=_passthrough_decorator,
            cdiv=lambda a, b: -(-a // b),
        )
    lang, stubbed = _ensure_module("triton.language")
    if stubbed:
        _set_default(lang, constexpr="constexpr")
    extra, stubbed = _ensure_module("triton.language.extra")
    if stubbed:
        _ensure_module("triton.language.extra.cann")
        cann, _ = _ensure_module("triton.language.extra.cann")
        ext, _ = _ensure_module("triton.language.extra.cann.extension")
        _set_default(cann, extension=ext)
        _set_default(
            ext,
            parallel=lambda *a, **k: range(a[1]),
            extract_slice=lambda *a, **k: None,
        )
        _ensure_parent_attr("triton.language", "extra")
        _ensure_parent_attr("triton.language.extra", "cann")
        _ensure_parent_attr("triton.language.extra.cann", "extension")

    # ---- tilelang ----
    tilelang, stubbed = _ensure_module("tilelang")
    if stubbed:
        _set_default(tilelang, PassConfigKey=_PassConfigKey, jit=_passthrough_decorator)
    _ensure_module("tilelang.tvm")
    tvm_mod, _ = _ensure_module("tilelang.tvm")
    _set_default(
        tvm_mod,
        tirx=types.SimpleNamespace(
            call_intrin=lambda *a, **k: None,
            op=types.SimpleNamespace(Op=type("Op", (), {"get": staticmethod(lambda name: ("op", name))})),
        ),
    )
    _ensure_parent_attr("tilelang", "tvm")
    _ensure_module("tilelang.ascend")
    lang_t, stubbed = _ensure_module("tilelang.ascend.language")
    if stubbed:
        _set_default(
            lang_t,
            macro=_passthrough_decorator,
            prim_func=_passthrough_decorator,
            Buffer=lambda shape, dtype: ("buf", shape, dtype),
            dynamic=lambda name: ("dyn", name),
        )
    _ensure_parent_attr("tilelang.ascend", "language")
    _ensure_module("tilelang.ascend.language.simd")
    _ensure_parent_attr("tilelang.ascend.language", "simd")
    _ensure_module("tilelang.carver")
    _ensure_module("tilelang.carver.arch")
    driver, _ = _ensure_module("tilelang.carver.arch.driver")
    _set_default(driver, get_num_cube_cores=lambda dev: 28)
    _ensure_parent_attr("tilelang.carver", "arch")
    _ensure_parent_attr("tilelang.carver.arch", "driver")
    layout, _ = _ensure_module("tilelang.layout")
    _set_default(
        layout,
        make_ascend_compact_nz_layout=lambda x: x,
        make_ascend_major_k_layout=lambda x: x,
    )
    profiler, _ = _ensure_module("tilelang.profiler")
    _set_default(profiler, do_bench=lambda *a, **k: None)

    # ---- tvm (dsa kernel 依赖 from tvm import tirx) ----
    tvm, stubbed = _ensure_module("tvm")
    if stubbed:
        _set_default(
            tvm,
            tirx=types.SimpleNamespace(
                call_intrin=lambda *a, **k: None,
                op=types.SimpleNamespace(Op=type("Op", (), {"get": staticmethod(lambda name: ("op", name))})),
            ),
        )


install()
