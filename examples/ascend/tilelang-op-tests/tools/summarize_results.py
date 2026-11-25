# -*- coding: utf-8 -*-
"""汇总逐用例测试日志: 统计打屏 + CSV / Markdown 结果表格.

输入: run_tests.sh 生成的 raw_results.tsv (列: order, nodeid, exit_code, seconds, log)
      与逐用例日志文件 (含 pytest 摘要行 / [perf] / [correctness] 打印行)
输出: <out-dir>/summary.csv    明细表格 (Excel 可直接打开, utf-8-sig)
      <out-dir>/summary.md     Markdown 表格
      终端: 总数/通过/失败等统计横幅 + 完整表格 + 失败用例清单
退出码: 0 = 无失败/错误; 1 = 存在失败或错误

用法 (由 run_tests.sh 自动调用, 也可单独执行):
    python tools/summarize_results.py --tsv results/<ts>/raw_results.tsv --out-dir results/<ts>
"""

import argparse
import csv
import pathlib
import re
import sys

# ---------------------------------------------------------------- 解析规则 --

RE_STATUS = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)")
RE_PERF = re.compile(r"\[perf\](?:\[(\w+)\])?[ \t]*(.+)")
RE_TL_MS = re.compile(r"\[perf\]\[tilelang\][^:\n]*:[ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*ms")
RE_CORRECTNESS = re.compile(r"\[correctness\][ \t]*(.*)")
RE_MAX_ABS = re.compile(r"max_abs(?:[_a-zA-Z]*)=([0-9.eE+-]+)")

STATUS_CN = {
    "PASS": "通过",
    "FAIL": "失败",
    "SKIP": "跳过",
    "ERROR": "错误",
    "TIMEOUT": "超时",
    "INTERRUPT": "中断",
    "NOT_FOUND": "未找到",
    "UNKNOWN": "未知",
}


def _op_name(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("gqa/"):
        return "GQA"
    if p.startswith("mha/"):
        return "MHA"
    if p.startswith("dsa/"):
        if "gbsa_baseline_bnsd" in p:
            return "DSA(GBSA基准-BNSD)"
        if "gbsa_baseline" in p:
            return "DSA(GBSA基准)"
        return "DSA"
    if p.startswith("mhc/"):
        return "mHC"
    if p.startswith("mixed_triton_tilelang/"):
        return "混合部署"
    return p.split("/")[0] if "/" in p else p


def _case_kind(function: str) -> str:
    if "perf" in function:
        return "性能"
    if "correctness" in function or "vector_mask" in function or "both_halves" in function:
        return "精度"
    return "功能"


def _shape_of(nodeid: str) -> str:
    if "[" in nodeid:
        return nodeid[nodeid.index("[") + 1 : nodeid.rindex("]")]
    return "—"


def _parse_outcome(log_text: str, exit_code: str) -> str:
    if exit_code == "124":
        return "TIMEOUT"
    if exit_code == "130":
        return "INTERRUPT"
    tail = log_text[-6000:]
    if "no tests ran" in tail:
        return "NOT_FOUND"
    passed = failed = skipped = errors = 0
    for num, word in RE_STATUS.findall(tail):
        num = int(num)
        if word == "passed":
            passed += num
        elif word == "failed":
            failed += num
        elif word == "skipped":
            skipped += num
        elif word in ("error", "errors"):
            errors += num
    if errors:
        return "ERROR"
    if failed:
        return "FAIL"
    if skipped and not passed:
        return "SKIP"
    if passed:
        return "PASS"
    # 摘要行缺失时按退出码兜底 (0=通过, 1=失败, 其他=错误)
    if exit_code == "0":
        return "PASS"
    if exit_code == "1":
        return "FAIL"
    return "UNKNOWN"


def _parse_perf(log_text: str) -> str:
    items = []
    for line in log_text.splitlines():
        if "[perf]" not in line:
            continue
        match = RE_PERF.search(line)
        if not match:
            continue
        tag = (match.group(1) or "").strip()
        text = match.group(2).strip()
        if not text or text.startswith("round "):
            continue  # 多轮明细行不入表, 只保留中位数/结论行
        entry = f"{tag}: {text}" if tag else text
        if entry not in items:
            items.append(entry)
    return " ; ".join(items)[:400]


def _parse_tilelang_ms(log_text: str) -> str:
    """提取统一的 tilelang kernel 耗时 (ms, 3 位小数); 无性能行时返回 '—'。"""
    match = RE_TL_MS.search(log_text)
    if match:
        return f"{float(match.group(1)):.3f}"
    return "—"


def _parse_precision(log_text: str, status: str, kind: str) -> str:
    max_abs = None
    for match in RE_MAX_ABS.finditer(log_text):
        value = float(match.group(1))
        if max_abs is None or value > max_abs:
            max_abs = value
    if kind == "性能":
        # 性能用例不做精度断言; GBSA 基准自带的交叉验证除外
        if RE_CORRECTNESS.search(log_text) and status == "PASS":
            return "PASS(交叉验证)" + (f" max_abs={max_abs:.3e}" if max_abs is not None else "")
        return "—"
    cell = status
    if max_abs is not None:
        cell += f"(max_abs={max_abs:.3e})"
    return cell


# ---------------------------------------------------------------- 主流程 --

def load_tsv(path: pathlib.Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, help="run_tests.sh 生成的 raw_results.tsv")
    parser.add_argument("--out-dir", required=True, help="结果输出目录")
    args = parser.parse_args()

    tsv_path = pathlib.Path(args.tsv)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_tsv(tsv_path)

    records = []
    for row in rows:
        nodeid = row["nodeid"]
        log_path = pathlib.Path(row["log"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        status = _parse_outcome(log_text, row["exit_code"])
        path, _, function = nodeid.partition("::")
        kind = _case_kind(function)
        records.append(
            {
                "序号": row["order"],
                "用例名称": nodeid,
                "算子名称": _op_name(path),
                "shape": _shape_of(nodeid),
                "用例类型": kind,
                "测试状态": status,
                "精度测试结果": _parse_precision(log_text, status, kind),
                "tilelang耗时(ms)": _parse_tilelang_ms(log_text) if kind == "性能" else "—",
                "性能测试数据": _parse_perf(log_text) if kind == "性能" else "—",
                "耗时(秒)": row["seconds"],
                "日志文件": str(log_path),
            }
        )

    # ---- 统计 ----
    total = len(records)
    by_status = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "TIMEOUT": 0, "INTERRUPT": 0, "NOT_FOUND": 0, "UNKNOWN": 0}
    by_kind_status = {}
    for record in records:
        by_status[record["测试状态"]] += 1
        key = (record["用例类型"], record["测试状态"])
        by_kind_status[key] = by_kind_status.get(key, 0) + 1
    total_seconds = sum(int(r["耗时(秒)"]) for r in records if str(r["耗时(秒)"]).isdigit())

    kinds = ("精度", "性能", "功能")
    bad_states = ("FAIL", "ERROR", "TIMEOUT", "INTERRUPT", "NOT_FOUND", "UNKNOWN")

    # ---- CSV ----
    csv_path = out_dir / "summary.csv"
    fields = ["序号", "用例名称", "算子名称", "shape", "用例类型", "测试状态", "精度测试结果", "tilelang耗时(ms)", "性能测试数据", "耗时(秒)", "日志文件"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # ---- Markdown ----
    def md_escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# 算子测试结果汇总\n\n")
        handle.write(f"- 总用例: {total},通过: {by_status['PASS']},失败: {by_status['FAIL']},"
                     f"跳过: {by_status['SKIP']},错误: {by_status['ERROR']},超时: {by_status['TIMEOUT']}\n")
        handle.write(f"- 总耗时: {total_seconds} 秒\n\n")
        header = "| " + " | ".join(fields[:-1]) + " |"
        separator = "|" + "---|" * (len(fields) - 1)
        handle.write(header + "\n" + separator + "\n")
        for record in records:
            cells = [md_escape(record[f]) for f in fields[:-1]]
            handle.write("| " + " | ".join(cells) + " |\n")

    # ---- 打屏 ----
    logs_hint = tsv_path.parent.parent / "logs" / tsv_path.parent.name
    line = "=" * 78
    print(line)
    print("                       算子测试汇总报告")
    print(line)
    print(f"  总用例数量 : {total}")
    print(f"  通过数量   : {by_status['PASS']}")
    print(f"  失败数量   : {by_status['FAIL'] + by_status['ERROR'] + by_status['TIMEOUT'] + by_status['INTERRUPT'] + by_status['NOT_FOUND'] + by_status['UNKNOWN']}"
          f"  (失败 {by_status['FAIL']} / 错误 {by_status['ERROR']} / 超时 {by_status['TIMEOUT']}"
          f" / 中断 {by_status['INTERRUPT']} / 未找到 {by_status['NOT_FOUND']} / 未知 {by_status['UNKNOWN']})")
    print(f"  跳过数量   : {by_status['SKIP']}")
    for kind in kinds:
        detail = ", ".join(
            f"{STATUS_CN[status]} {by_kind_status.get((kind, status), 0)}"
            for status in ("PASS", "FAIL", "SKIP", "ERROR", "TIMEOUT", "INTERRUPT")
            if by_kind_status.get((kind, status), 0)
        )
        count = sum(v for (k, _), v in by_kind_status.items() if k == kind)
        print(f"  {kind}用例   : {count} 条  ({detail})" if detail else f"  {kind}用例   : {count} 条")
    print(f"  总耗时     : {total_seconds} 秒")
    print("-" * 78)
    print(f"  明细表格   : {csv_path}")
    print(f"              {md_path}")
    print(f"  逐用例日志 : {logs_hint}")
    print(line)

    # 失败/错误用例清单
    bad = [r for r in records if r["测试状态"] in bad_states]
    if bad:
        print("\n未通过用例清单:")
        for record in bad:
            print(f"  [{record['测试状态']}] {record['用例名称']}  (日志: {record['日志文件']})")
        print()

    # 完整表格打屏
    print("\n" + header)
    print(separator)
    for record in records:
        cells = [md_escape(record[f]) for f in fields[:-1]]
        print("| " + " | ".join(cells) + " |")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
