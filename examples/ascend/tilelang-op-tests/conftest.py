# -*- coding: utf-8 -*-
"""根 conftest: ``standalone/`` 目录不参与 pytest 目录扫描收集。

- ``tools/generate_testcases.py`` 的全量收集因此不会把 ``standalone/`` 下的
  独立测试写进回归清单 ``testcases.txt``, ``tools/run_tests.sh`` 全量回归
  同样不会执行它们;
- 显式指定路径执行不受影响 (pytest 对命令行显式传入的文件始终收集):
  ``pytest standalone/gemm/test_gemm_general.py -v``
"""

collect_ignore = ["standalone"]
