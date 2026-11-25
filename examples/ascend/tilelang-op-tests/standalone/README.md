# standalone —— 独立单算子测试

存放**独立运行**的单算子测试脚本, 与回归套件 (`gqa/` `mha/` `dsa/` `mhc/`
`mixed_triton_tilelang/`) 相互隔离:

- 根 `conftest.py` 声明 `collect_ignore = ["standalone"]`,
  `tools/generate_testcases.py` 的全量收集不含本目录, `testcases.txt`
  回归清单与 `tools/run_tests.sh` 全量回归均不涉及;
- 需要执行时**显式指定文件路径** (pytest 对命令行显式传入的文件始终收集,
  不受目录扫描忽略影响)。

## gemm/test_gemm_general.py

`T.gemm` (TileLang Ascend 语言层) 泛化测试, 34 条用例:
L1(NT) / L0(NN/NT/TN/TT) × fp16/bf16/fp32/int8/fp8e4m3fn ×
对齐/非对齐形状 × 输出类型 × `clear_accum` 累加;
参考实现为 CPU float64 (`op(A) @ op(B)` 按输出类型舍入)。

```bash
# Ascend 服务器 (tilelang + torch_npu + CANN 环境就绪)
python -m pytest standalone/gemm/test_gemm_general.py -v
```

也可整文件拷贝到 tilelang 仓库 `testcases/functional/` 下按原路径运行。

### 设备契约 (输入必须在 NPU 上)

ascend kernel 经 `tilelang.compile(..., target="ascend", out_idx=...)`
编译后, host stub 会把**第一个 torch.Tensor 输入**作为 allocator anchor,
其设备类型必须等于编译目标 (ascend = DLPack `kDLExtDev` = 12, 与 torch_npu
的 NPU 张量一致), 输出也在 anchor 所在设备分配。传 CPU 张量 (`kDLCPU` = 1)
会报:

```
RuntimeError: kernel: allocator anchor has the wrong device type; expected: 1, got: 12
```

脚本内已处理: 导入 torch_npu 注册 npu 设备 → CPU 固定种子生成输入 →
搬运到 NPU 喂 kernel → `torch.npu.synchronize()` → 回 CPU 与 float64
参考比较 (Ascend 不支持 fp64 matmul)。
