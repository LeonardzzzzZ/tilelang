# -*- coding: utf-8 -*-
"""生成/维护用例清单 testcases.txt: 收集仓库内全部 pytest 用例节点 ID.

用法 (在仓库根目录):
    python tools/generate_testcases.py                    # 生成 ./testcases.txt (全量清单)
    python tools/generate_testcases.py -o other.txt       # 指定输出
    python tools/generate_testcases.py --sync my.txt      # 把仓库新增用例追加到自维护清单

testcases.txt 的使用方式:
    - 供 tools/run_tests.sh 逐条执行; 每行一个 pytest 节点 ID
      (相对仓库根目录, 如 gqa/test_gqa.py::test_x[param]);
    - 空行与 '#' 开头的注释行会被执行脚本忽略 —— 不想执行某条用例时整行注释或删除,
      想补充用例时直接追加一行合法节点 ID 即可;
    - 推荐把清单复制到 run_tests.sh 所在目录作为 "外部清单" 自由维护
      (外部清单优先级高于仓库内置清单, 且不受 git 更新影响);
    - 仓库新增用例后, 用 --sync <你的清单> 把新用例追加进去
      (只追加不存在的用例, 保留你的所有增删与注释)。

原理: 运行 `pytest --collect-only -q` 并附带 tools._collect_stubs 插件
(依赖缺失时才 stub, 依赖齐全的服务器上为真实收集), 逐行解析节点 ID。
"""

import argparse
import pathlib
import subprocess
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

HEADER_TEMPLATE = """# tilelang-op-tests 用例清单 (由 tools/generate_testcases.py 生成)
# 生成时间: {timestamp}
# 用例总数: {total}
{module_counts}#
# 使用说明 (本文件可直接编辑):
#   - 每行一个 pytest 节点 ID; 空行与 '#' 开头的注释行不执行;
#   - 不想执行某条用例: 整行注释 (行首加 '#') 或删除该行;
#   - 想补充用例: 直接追加一行合法节点 ID;
#   - 建议复制本文件到 run_tests.sh 所在目录维护 (外部清单优先, 不受 git 更新影响);
#   - 仓库新增用例后: python tools/generate_testcases.py --sync <你的清单> 追加新用例。
#
"""


def collect_nodeids() -> list:
    cmd = [
        sys.executable, "-m", "pytest",
        "--collect-only", "-q",
        "-p", "tools._collect_stubs",
        str(REPO_ROOT),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:  # 收集报错时仍尝试解析已收集部分, 但提示告警
        print(f"[warn] pytest --collect-only 退出码 {proc.returncode}, 尝试解析可用部分", file=sys.stderr)
    nodeids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("=") and not line.startswith("warnings summary"):
            nodeids.append(line)
    seen = set()
    return [nid for nid in nodeids if not (nid in seen or seen.add(nid))]


def write_full_list(path: pathlib.Path, nodeids: list) -> None:
    counts = Counter(
        "/".join(nid.replace("\\", "/").split("::")[0].split("/")[:2]) for nid in nodeids
    )
    module_lines = "".join(f"#   {module}: {count}\n" for module, count in sorted(counts.items()))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            HEADER_TEMPLATE.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total=len(nodeids),
                module_counts=module_lines,
            )
        )
        handle.write("\n")
        for nid in nodeids:
            handle.write(nid + "\n")


def sync_user_list(path: pathlib.Path, nodeids: list) -> int:
    """把仓库新增用例追加到用户自维护清单; 保留用户全部已有行 (含注释行)。

    已存在(无论生效还是被注释)的用例不重复追加; 用户清单中已不在仓库的
    生效用例视为失效, 打屏提示但不改动。
    """
    if not path.exists():
        print(f"[error] 清单不存在: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    present = set()
    for line in lines:
        active = line.strip()
        while active.startswith("#"):
            active = active.lstrip("#").lstrip()
        if "::" in active:
            present.add(active)

    repo_set = set(nodeids)
    new_ids = [nid for nid in nodeids if nid not in present]
    stale = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#") and "::" in line and line.strip() not in repo_set
    ]

    if new_ids:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n# ---- 以下为自动同步追加的新增用例 (--sync, 时间: ")
            handle.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ") ----\n")
            for nid in new_ids:
                handle.write(nid + "\n")

    print(f"[sync] {path}: 仓库用例 {len(nodeids)} 条, 新增追加 {len(new_ids)} 条, 失效 {len(stale)} 条")
    for item in stale:
        print(f"    [stale] {item}  (仓库中已不存在, 可删除或注释该行)")
    if not new_ids and not stale:
        print("[sync] 无新增、无失效, 清单已是最新")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="testcases.txt", help="全量清单输出文件 (默认 ./testcases.txt)")
    parser.add_argument("--sync", metavar="USER_LIST", help="把仓库新增用例追加到该自维护清单 (不覆盖已有内容)")
    args = parser.parse_args()

    nodeids = collect_nodeids()
    if not nodeids:
        print("[error] 未收集到任何用例, 请检查测试文件与依赖", file=sys.stderr)
        return 1

    if args.sync:
        return sync_user_list(pathlib.Path(args.sync), nodeids)

    output = pathlib.Path(args.output)
    write_full_list(output, nodeids)
    print(f"[generate] 共收集 {len(nodeids)} 条用例, 已写入 {output}")
    counts = Counter(
        "/".join(nid.replace("\\", "/").split("::")[0].split("/")[:2]) for nid in nodeids
    )
    for module, count in sorted(counts.items()):
        print(f"    {module}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
