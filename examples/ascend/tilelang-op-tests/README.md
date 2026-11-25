# tilelang-op-tests

TileLang 算子测试用例脚本集(Ascend NPU,pytest)。

本仓库专门存放各算子的 **shape 泛化测试脚本**:本地/仓库侧只负责脚本编写,
实际编译与执行在配备 Ascend NPU(CANN + torch_npu + tilelang)的服务器上进行。

## 目录结构

```text
tilelang-op-tests/
├── README.md
├── .gitignore
├── testcases.txt           # 全部 pytest 用例节点 ID 清单 (124 条, tools 生成)
├── tools/                  # 端到端测试框架
│   ├── run_tests.sh        # 主脚本: 克隆仓库 + 逐条执行 + 汇总 (拷到服务器单独运行)
│   ├── generate_testcases.py     # 收集用例生成 testcases.txt
│   ├── summarize_results.py      # 汇总日志 → 统计打屏 + CSV/Markdown 表格
│   └── _collect_stubs.py         # 收集期依赖 stub 插件 (真实导入优先, 缺失才 stub)
├── gqa/                    # GQA (Grouped-Query Attention) 算子
│   └── test_gqa.py         # 自包含 pytest 脚本: kernel + 参考实现 + 用例表 + 性能计时
├── mha/                    # MHA (Multi-Head Attention) 算子
│   └── test_mha.py         # 自包含 pytest 脚本 (支持静态/动态 shape)
├── dsa/                    # DSA (GLM DSA, DeepSeek Sparse Attention) 算子
│   ├── test_dsa.py         # 自包含 pytest 脚本 (稀疏注意力, 含 SIMD mask 探针)
│   ├── test_dsa_gbsa_baseline.py       # GBSA 性能基准 (TND, CANN 9.2.0 实测可用)
│   └── test_dsa_gbsa_baseline_bnsd.py  # GBSA 基准 BNSD 零拷贝变体 (需新版 CANN, 旧版优雅 skip)
├── mhc/                    # mHC (流形约束超连接) 算子 (多文件 kernel 结构)
│   ├── mhc_pre.py          # Pre 入口: 校验/分发/缓存 (MK 与 Whole-M 双实现)
│   ├── mhc_pre_mk.py       # Pre MK 实现 + 共享 SIMD 宏 (逐字复制自上游)
│   ├── mhc_pre_large.py    # Pre Whole-M 实现 (逐字复制自上游)
│   ├── mhc_post.py         # Post 入口 + kernel (逐字复制自上游)
│   └── test_mhc.py         # 集成 pytest: 路由 + 正确性 + Pre/Post 集成 + 性能
├── mixed_triton_tilelang/  # triton-ascend 与 tilelang 混合部署验证
│   ├── test_mixed_deployment.py   # pytest 入口 (同进程 + 跨进程场景)
│   ├── triton_matmul.py           # triton 侧 matmul (官方教程 03 复制) + 校验入口
│   ├── worker_tilelang.py         # 独立进程 worker: 仅执行 tilelang GQA
│   └── worker_triton.py           # 独立进程 worker: 仅执行 triton matmul
└── standalone/             # 独立单算子测试 (不入回归清单, 显式路径执行, 见 standalone/README.md)
    └── gemm/
        └── test_gemm_general.py   # T.gemm 泛化测试 (L1/L0 × dtype × 转置组合)
```

> 后续新增算子时,按 `gqa/`、`mha/` 的模式新增 `<算子名>/test_<算子名>.py` 即可;
> 新增/修改用例后运行 `python tools/generate_testcases.py` 更新 `testcases.txt`。## 端到端测试框架(tools)

在内部服务器通过 git 拉取代码完成精度与性能用例的全量回归:

```bash
# 1. 把 tools/run_tests.sh 单独拷贝到服务器任意目录, 然后:
bash run_tests.sh                                   # 默认克隆 gitcode 仓库并全量执行
GIT_TOKEN=<令牌> bash run_tests.sh                  # 私有仓需要令牌 (GIT_USER 可选)
PYTHON=/path/to/python bash run_tests.sh            # 指定解释器 (conda/venv 等)

# 可选环境变量:
#   REGENERATE_CASES=1   在服务器上重新收集用例清单 (默认用仓库内置 testcases.txt)
#   CASE_TIMEOUT=1800    单条用例超时秒数
#   EXTRA_PYTEST_ARGS    附加 pytest 参数
#   REPO_URL             使用内部镜像地址
```

产物(均在脚本所在目录):

| 路径 | 内容 |
|---|---|
| `testcases.txt` | **可编辑执行清单**(首次运行自动从仓库内置清单复制到脚本目录;之后自由增删) |
| `logs/<时间戳>/NNN_<用例>.log` | 逐用例执行日志 |
| `results/<时间戳>/raw_results.tsv` | 原始记录(nodeid/退出码/耗时/日志路径) |
| `results/<时间戳>/summary.csv` | **明细表格**(Excel 直接打开):用例名称/算子名称/shape/精度测试结果/tilelang耗时(ms)/性能测试数据 |
| `results/<时间戳>/summary.md` | 同内容的 Markdown 表格 |

**用例清单的增删维护**(核心机制):

- 首次运行后,脚本目录会出现一份 `testcases.txt` 副本——**之后所有执行都以它为准**,
  git 更新不会覆盖你的修改;
- **不执行某条用例**:整行注释(行首加 `#`)或删除该行;
- **补充用例**:直接追加一行合法的 pytest 节点 ID;
- **重置**:删除脚本目录的 `testcases.txt`(下次运行会重新从仓库内置清单复制);
- **同步仓库新增用例**(保留你的增删与注释,只追加不存在的用例):
  `python tools/generate_testcases.py --sync testcases.txt` 或
  `REGENERATE_CASES=1 bash run_tests.sh`;
- 清单中已失效的用例(仓库中已不存在)会在执行时记为 `NOT_FOUND`,sync 时也会提示。

打屏输出总用例数/通过/失败/跳过统计及精度/性能/功能分类明细;
退出码 0=全部通过(含跳过),1=存在失败或错误,可接入 CI 门禁。


## 环境要求(服务器)

- Python >= 3.9
- tilelang(含 `tilelang.ascend` 模块)
- PyTorch + torch_npu(CANN 环境)
- triton-ascend(混合部署验证所需,`pip install triton-ascend`,参考 3.2.1 / CANN 9.0.0)
- 当前测试卡:28 个 cube 核心

## 运行方式

```bash
pytest gqa/test_gqa.py -v -s              # GQA 全部用例 (correctness + perf)
pytest gqa/test_gqa.py -v -k "not perf"   # 只跑正确性
pytest gqa/test_gqa.py -v -k "s1-7168"    # 按 case id 过滤 shape

pytest mha/test_mha.py -v -s              # MHA 全部用例 (静态 + 动态 + perf)
pytest mha/test_mha.py -v -k "dyn"        # 只跑动态 shape 用例
pytest mha/test_mha.py -v -k "q4097"      # 按 shape 过滤

pytest dsa/test_dsa.py -v -s              # DSA 全部用例 (回归 + 探针 + perf)
pytest dsa/test_dsa.py -v -k "not perf"   # 跳过性能
pytest dsa/test_dsa.py -v -k "vector_mask"  # 只跑 SIMD mask 探针

pytest dsa/test_dsa_gbsa_baseline.py -v -s   # DSA vs GBSA 性能基准 (需 cann_ops_transformer)
pytest dsa/test_dsa_gbsa_baseline_bnsd.py -v -s   # BNSD 零拷贝变体 (需新版 CANN 支持 BNSD)

pytest mhc/test_mhc.py -v -s                # mHC 全部用例 (路由 + 正确性 + 性能)
pytest mhc/test_mhc.py -v -k "routing"      # CPU 路由测试 (无需 NPU)
pytest mhc/test_mhc.py -v -k "not perf"     # 跳过性能用例

pytest mixed_triton_tilelang/test_mixed_deployment.py -v -s            # 混合部署全部场景
pytest mixed_triton_tilelang/test_mixed_deployment.py -v -k same_process
pytest mixed_triton_tilelang/test_mixed_deployment.py -v -k cross_process

pytest standalone/gemm/test_gemm_general.py -v                # 独立单算子测试 (不入回归清单)
```

## GQA 算子说明

- kernel 实现来自 [tilelang-asc](https://github.com/HelloWorld07-7/tilelang-asc)
  仓库 `examples/ascend/flash_attention/`(commit 77ced9c),`test_gqa.py`
  为不依赖原工程的自包含整合版(kernel 逐字内嵌,便于同步上游更新)。
- shape 泛化硬性约束(由 kernel 决定):
  - `D = 128` 固定;
  - `q_len = S1 * G` 与 `S2` 必须能被 128 整除;
  - `num_blocks` 必须整除 M tiles 数(`q_len / 128`);
  - 28 核满载最优配置 `num_blocks = 28`,要求 `q_len % 3584 == 0`。
- 测试规格:`dtype` 与 `out_dtype` 均固定为 **BF16**,`manual_schedule=False`(自动调度),
  精度容差 `atol = rtol = 1.6e-2`,均由用例默认值承载(不在测试函数里写死),
  需变更时在单个 case 中显式指定。
- 正确性判据(逐元素):`|out - ref| <= atol + rtol * |ref|`(对比 torch SDPA)。

## MHA 算子说明(mha)

- kernel 实现来自 [dangyw18/tilelang](https://github.com/dangyw18/tilelang/tree/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/)
  (commit 49a5122),`test_mha.py` 为不依赖原工程的自包含整合版。
  **与 `gqa/` 的 kernel 是不同版本**(本版支持动态 shape、Q 每 KV 迭代重载 L0A、
  softmax 先缩放写回、无 unit_flag_ctrl),两份 kernel 独立维护、勿混用。
- 约束(由 kernel 决定):
  - `D = 128` 固定;`kv_len` 必须被 128 整除(动态/静态均要求);
  - **静态**编译时 `q_len` 也必须被 128 整除;**动态**(`T.dynamic`,一次编译多 shape
    复用)时 `q_len` 允许非对齐(如 4097,GM 尾部 clamp + L1 零填充);
  - `num_blocks` 显式指定要求静态 q_len 且整除 M tiles(`ceil(q_len/128)`);
    默认 `None` = 每 core 一个 M tile(M tiles=28 的 shape 天然用满 28 核)。
- 测试规格:`dtype=bfloat16`、`out_dtype=float32`(fork MHA 口径)、
  `atol = rtol = 5e-2`(MHA 规格,对齐 fork `rel<0.05` 判据口径),
  均由用例默认值承载,需变更时在单个 case 中显式指定。
- 性能计时参数采用 fork 口径(`_n_warmup=30, _n_repeat=500`,msprof_detail),
  可选 torch SDPA 基准对比(`MHA_BENCH_TORCH=0` 关闭)。

### 用例分布(mha)

| 分组 | case |
|---|---|
| 静态·小 shape | `mha_static_q128-k128`、`mha_static_q256-k256`、`mha_static_q1024-k1024` |
| 静态·基准 | `mha_static_q4096-k4096`(fork 原始基准,M tiles=32) |
| 静态·28 核适配 | `mha_static_q3584-k3584`(M tiles=28,默认调度天然 28 核)、`mha_static_q7168-k7168_nb28`(M tiles=56,每 block 2 tiles) |
| 动态 shape | `mha_dyn_q4096-k128_q4097-k128_q4097-k256`(含非对齐 q_len,来自 fork test_mha)、`mha_dyn_q4096-k4096_q7168-k7168`(大 shape) |

## DSA 算子说明(dsa)

- kernel 实现来自 [HhpPCc007/tilelang-asc](https://github.com/HhpPCc007/tilelang-asc/blob/3c34da6ee7bd102f91b85f2fa46bdc5e71b0b96a/examples/ascend/glm_dsa/core.py)
  (commit 3c34da6),`test_dsa.py` 为不依赖原工程的自包含整合版。
- 输入契约(kernel 固定属性:64 头 / head_dim=256 / scale=0.0625 / TopK 上限 2048):
  - Q/K/V `[B,64,sq 或 T,256]` BF16(每头独立);indices `[B,sq,Ksel]` int32
    (`Ksel=min(2048,T)`,唯一有效位置,无 -1);mask `[B,1,sq,T]` bool;
    O `[B,sq,64,256]` BF16
- shape 约束极宽松:B/sq/T 只需为正(非对齐长度如 T=96/160/257/2049 走标量回退路径);
  `num_blocks` 默认自动取 min(cube 核数, 任务数)(28 核卡上 ≤ 28),
  显式指定须满足 `0 < nb <= B·sq·(64/heads_per_task)`;`heads_per_task` 默认 4。
- 测试规格:`dtype=out_dtype=BF16`(kernel 固定),FP32 golden,
  `atol = rtol = 1e-2`(fork test 口径),均由用例默认值承载。
- 特殊回归(fork 同款):全 mask 输出全零(非 NaN)、单 tile 双 AIV 半区贡献(0.5 均值)、
  SIMD vector-mask 探针(kv=256/2048/8192,精确 0/1 flag)。
- 性能计时口径:msprof_detail,30 预热 / 100 重复;flops = 4·B·sq·64·256·Ksel;
  DSA 无直接等价 torch 算子,不附带 torch 基准(fork 以 GBSA 为对照,不在本仓库内)。
- 注意:`reference_attention` 为 Python 逐 query 循环的 FP32 golden,大 shape(如 sq=448)较慢属预期。

### 用例分布(dsa)

| 分组 | case |
|---|---|
| 单 TopK block(Ksel≤128) | `dsa_b1_sq32_t96_nb8`(mixed / all_keep) |
| 多 TopK block | `dsa_b1_sq32_t1024_nb8`、`dsa_b1_sq16_t2048_nb8`(perf) |
| batch 维 | `dsa_b2_sq32_t512_nb16` |
| 全 mask→全零 | `dsa_b1_sq32_t512_nb8_all_drop` |
| 残块尾 / 非对齐 T | `dsa_b1_sq16_t160_nb8`(Ksel=160)、`dsa_b1_sq8_t257_nb8`、`dsa_b1_sq4_t2049_nb4`(TopK 上限) |
| 自定义 head 分组 | `dsa_b1_sq16_t2048_hpt32`、`dsa_b1_sq17_t2049_hpt2`、`dsa_b1_sq1_t257_hpt64` |
| 任务数不整除核数 | `dsa_b1_sq3_t257_nb28` |
| README 性能参考 | `dsa_b1_sq448_t4096`(perf) |
| 独立用例 | `test_both_halves_of_a_tile_contribute`、`test_vector_mask_flags[256/2048/8192]` |

### GBSA 性能基准(dsa/test_dsa_gbsa_baseline.py)

对标 [ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/attention/generic_block_sparse_attention)
的 GBSA 算子(主算子 + [metadata 前置算子](https://gitcode.com/cann/ops-transformer/tree/master/attention/generic_block_sparse_attention_metadata)):

- **依赖**:`cann_ops_transformer` 包(ops-transformer 仓库安装;fork 验证用 commit
  `aaad76f2`),未安装时优雅 skip,不影响 DSA 主套件;
- **口径**:GBSA 侧被计时内容 = 索引适配(mask 折叠,pack_indices 口径)+ metadata 生成 +
  主算子(完整调用链,输入在 NPU);tilelang 侧 = kernel 调用;ratio = GBSA / TileLang;
- **语义对齐**:GBSA 侧使用 TND 布局(实测服务器 CANN 9.2.0 的 metadata 算子仅支持
  TND qInputLayout,BNSD 报 EZ1001),Q/K/V 由 DSA 的 `[B,H,S,D]` 转为
  `(B·S,H,D)/(B·T,H,D)` 并配 `cu_seqlens` 前缀和(转换计入 GBSA 侧计时);
  `block_shape=[1,1]` token 级稀疏对齐 per-position top-k;逐位置 mask 折叠进稀疏
  索引(升序、唯一、前 count 位);
- **基准点**:`sq=16/T=2048`、`B=2/sq=32/T=512`、`sq=17/T=2049`、`sq=448/T=4096`
  (fork README 测量 shape;`sq=1/T=4097`、`sq=8192/T=8192` 注释中按需打开);
- **计时参数**:GBSA 5 预热/20 重复(完整调用链),tilelang 30/100;backend 由
  `DSA_BENCH_BACKEND` 控制(默认 msprof_detail,`wall` 为端到端口径),两侧一致;
- **交叉验证**:双侧输出均对 FP32 golden 校验(大 shape 采样 16 行,atol=rtol=1e-2),
  同时验证适配逻辑正确性;
- **多轮验收**:`DSA_GBSA_ROUNDS=3` 多轮 A/B 交替测量取中位数并输出范围;
  `DSA_GBSA_ASSERT_FASTER=1` 断言 ratio > 1(tilelang 必须更快),用于正式验收;
- **BNSD 变体**(`test_dsa_gbsa_baseline_bnsd.py`):保留"零拷贝"原方案——DSA 的
  Q/K/V 即 GBSA 的 BNSD 布局,原张量直传(无 TND 布局转换开销),供支持 BNSD
  qInputLayout 的新版 CANN 验证;仅支持 TND 的 CANN 上优雅 skip(识别 EZ1001);
  与主基准共用基准点表、mask 折叠与多轮计时逻辑;
- `DSA_GBSA_BASELINE=0` 可强制跳过全部基准用例。

## mHC 算子说明(mhc)

- kernel 来自 [yangsichan/tilelang-asc](https://github.com/yangsichan/tilelang-asc/tree/053ddb470aefcbc6972d994739983734eeb4eaf8/examples/ascend/mhc)
  (commit 053ddb4),**多文件结构**(Pre 入口 + MK/Whole-M 双实现 + Post),
  已逐字节复制到本目录;唯一改动是 `mhc_pre.py` / `mhc_pre_large.py` 的包内相对导入
  改为平铺导入(kernel 实现零改动,同步上游时整文件替换)。
- 契约:`H ∈ {4096, 5120, 7168}`,N 取值 1..8192(serving 范围);
  Pre 按 N 自动在 MK(`N ≤ 2304` 等)与 Whole-M(`N ≥ 3585` 等)间分发,
  中间区域按 MK tile 完整性分发;容量默认 **28 AIC / 56 AIV**(与测试卡一致)。
- 精度规格(fork 开发口径):`layer_input`(BF16)atol=0.03125/rtol=1e-3;
  FP32 输出(pre/post/comb)atol=5e-5/rtol=1e-3;Post 输出(BF16)atol=0.03125/rtol=1e-3。
- 性能:msprof_detail,30 预热/500 测量(fork bench 口径),计时含每次调用的
  新鲜输出分配(Pre 还含 MK workspace 分配);`MHC_BENCH_BACKEND` 可换口径。
- Pre 首次调用会 JIT 编译一个 C++ 分配扩展(需 Ninja 与 C++ 编译器,上游要求一致)。

### 用例分布(mhc)

| 分组 | case |
|---|---|
| CPU 路由(无需 NPU) | mk/large 分发边界(2304/2305/2400/2401 等 10 组)、非默认核数 6 组、非法几何 6 组 |
| Pre 正确性 | `pre_h4096_n16`(冒烟)、`pre_h4096_n1_zero`(零输入 RMS)、`pre_h5120_n33`(残块尾)、`pre_h7168_n129`(系数批次)、`pre_h4096_n2305`(large 空 AIV 半区)、`pre_h5120_n2400`(m240 完整)、`pre_h7168_n2401`、`pre_h4096_n3137_flat`([N,4] 形态)、`pre_h5120_n3584`(m256 完整)、`pre_h7168_n3585`(自然边界)、`pre_h4096_n8192`(perf) |
| Post 正确性 | 单 token、持久循环尾、双隐藏 tile、批处理、小核数(c4)、`post_h4096_b4096`(perf) |
| 专项回归 | bind 换流执行 + 原地参数更新 + 新鲜输出、缓存接受新输入对象、非法输入拒绝(Pre/Post 各 2 组)、**Pre/Post 同流集成** |

## 用例分布(gqa)

| 分组 | case |
|---|---|
| 冒烟/小 shape | `s1-128_g-1_s2-128`、`s1-128_g-2_s2-256`、`s1-256_g-2_s2-512` |
| 中 shape | `s1-1024_g-8_s2-1024`、`s1-512_g-4_s2-1024`(非方阵) |
| 28 核专项(num_blocks=28) | `s1-128_g-28_s2-1024`(1 tile/核)、`s1-896_g-32_s2-2048`(8 tiles/核)、`s1-8192_g-28_s2-8192`、`s1-7168_g-32_s2-8192`、`s1-7168_g-32_s2-7168`(方形,均 64 tiles/核) |
| 上游回归参考(num_blocks=32) | `s1-8192_g-32_s2-8192` |

## 混合部署验证说明(mixed_triton_tilelang)

验证 **triton-ascend 算子**与 **tilelang 算子**在同进程/跨进程部署形态下的共存性
(功能正确性,不含性能对比):

- **triton 侧**:matmul kernel 复制自 triton-ascend 官方教程
  `third_party/ascend/tutorials/03-matrix-multiplication.py`(fp16 输入/fp32 累加,
  含 `extension.parallel` 双 vector core 写出与 autotune),判据 `atol = rtol = 1e-3`;
- **tilelang 侧**:复用 `gqa/test_gqa.py` 的 GQA kernel(小 shape `S1=128, G=2, S2=256`,
  进程内共享编译缓存),判据与 GQA 规格一致(`atol = rtol = 1.6e-2`);
- **场景矩阵**:

| 场景 | 用例 | 验证点 |
|---|---|---|
| 同进程·导入 | `test_same_process_import_order` | 双框架双向导入顺序(子进程) |
| 同进程·先后 | `test_same_process_tilelang_then_triton` / `test_same_process_triton_then_tilelang` | 先后执行双向顺序,双方结果正确 |
| 同进程·交替 | `test_same_process_interleaved` | 多轮 GQA↔matmul 交替,无状态串扰 |
| 跨进程·顺序 | `test_cross_process_sequential` | 两个 worker 进程先后运行(双向) |
| 跨进程·并发 | `test_cross_process_concurrent` | 两个 worker 同卡并行,资源不冲突 |

- **worker 约定**:独立可执行(`python mixed_triton_tilelang/worker_*.py`),
  成功输出 `WORKER_PASS` 且退出码 0,失败输出 `WORKER_FAIL` 且退出码 1;
- **超时控制**:环境变量 `MIXED_WORKER_TIMEOUT`(秒,默认 900),编译较慢时调大;
- 需替换为实际 triton 算子时,改 `triton_matmul.py` 并保持
  `run_triton_matmul_check` 接口不变,上层用例无需改动。
