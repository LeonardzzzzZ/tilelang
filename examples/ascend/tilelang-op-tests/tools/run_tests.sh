#!/usr/bin/env bash
# ============================================================================
# tilelang-op-tests 端到端测试框架主脚本 (拷贝本脚本到内部服务器单独运行)
#
# 用法:
#   bash run_tests.sh                 # 克隆/更新仓库并执行 testcases.txt 全部用例
#
# 可配环境变量 (均有默认值):
#   REPO_URL           仓库地址 (本副本默认 github 上的 LeonardzzzzZ/tilelang;
#                      独立仓布局: gitcode 上的 tilelang-op-tests)
#   GIT_TOKEN          访问令牌 (私有仓需要; 会注入 https 克隆 URL)
#   GIT_USER           https 认证用户名 (默认 oauth2; PAT 作为密码)
#   REPO_BRANCH        克隆/更新的分支 (本副本默认 tilelang-op-tests;
#                      独立仓布局用 main)
#   SUITE_REL_PATH     测试套件相对仓库根的路径 (独立仓默认为空; github 嵌套
#                      布局为 examples/ascend/tilelang-op-tests)
#   PYTHON             python 解释器 (默认 python3; 有 conda/venv 时按需指定)
#   TESTCASES_FILE     用例清单文件名 (默认 testcases.txt); 查找优先级:
#                      绝对路径 > 脚本所在目录(用户自维护, 增删/注释#均生效,
#                      不受 git 更新影响, 首次运行自动创建副本) > 仓库内置全量清单
#   CASE_TIMEOUT       单条用例超时秒数 (默认 1800)
#   EXTRA_PYTEST_ARGS  附加 pytest 参数 (默认空)
#   REGENERATE_CASES   =1 时重新收集全量用例, 并把新增用例同步进外部清单
#
# 产物 (均在脚本所在目录下):
#   logs/<时间戳>/NNN_<用例>.log      逐用例日志
#   results/<时间戳>/raw_results.tsv  原始执行记录 (nodeid/退出码/耗时/日志)
#   results/<时间戳>/summary.csv      明细表格 (Excel 可直接打开)
#   results/<时间戳>/summary.md       Markdown 表格
#   退出码: 0 = 全部通过(含跳过); 1 = 存在失败/错误; 130 = 被 Ctrl+C 中断
#
# 中断行为: 运行中按 Ctrl+C 会立即终止当前用例 (对 pytest 及其编译/算子子进程
#   执行 INT->TERM->KILL 升级终止 —— pytest 自身会吞掉第一次 SIGINT, 直接前台
#   运行时表现为 "按 Ctrl+C 无法终止", 本脚本已规避), 并对已完成用例出汇总;
#   再按一次 Ctrl+C 立即退出不汇总。
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="tilelang-op-tests"
REPO_DIR="${SCRIPT_DIR}/${REPO_NAME}"
REPO_URL="${REPO_URL:-https://github.com/LeonardzzzzZ/tilelang.git}"
GIT_TOKEN="${GIT_TOKEN:-}"
GIT_USER="${GIT_USER:-oauth2}"
PYTHON="${PYTHON:-python3}"
TESTCASES_FILE="${TESTCASES_FILE:-testcases.txt}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1800}"
EXTRA_PYTEST_ARGS="${EXTRA_PYTEST_ARGS:-}"
REGENERATE_CASES="${REGENERATE_CASES:-0}"
# 布局适配 (本副本嵌在 tilelang 大仓中):
#   本布局 (github fork):     REPO_BRANCH=tilelang-op-tests,
#                            SUITE_REL_PATH=examples/ascend/tilelang-op-tests
#   独立仓布局 (gitcode):     REPO_BRANCH=main, SUITE_REL_PATH="" (套件即仓库根)
REPO_BRANCH="${REPO_BRANCH:-tilelang-op-tests}"
SUITE_REL_PATH="${SUITE_REL_PATH:-examples/ascend/tilelang-op-tests}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/${TS}"
RESULT_DIR="${SCRIPT_DIR}/results/${TS}"
RAW_TSV="${RESULT_DIR}/raw_results.tsv"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

log() { echo "[run_tests $(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------- 1. 代码仓
clone_url="${REPO_URL}"
if [ -n "${GIT_TOKEN}" ]; then
    case "${REPO_URL}" in
        *"@"*) ;;
        https://*) clone_url="${REPO_URL/https:\/\//https:\/\/${GIT_USER}:${GIT_TOKEN}@}" ;;
        http://*) clone_url="${REPO_URL/http:\/\//http:\/\/${GIT_USER}:${GIT_TOKEN}@}" ;;
        *) clone_url="https://${GIT_USER}:${GIT_TOKEN}@${REPO_URL}" ;;
    esac
fi

if [ -d "${REPO_DIR}/.git" ]; then
    log "仓库已存在, 更新: ${REPO_DIR} (分支 ${REPO_BRANCH})"
    git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}" 2>/dev/null || log "[warn] git fetch 失败"
    git -C "${REPO_DIR}" checkout "${REPO_BRANCH}" 2>/dev/null || log "[warn] 切换到分支 ${REPO_BRANCH} 失败"
    if git -C "${REPO_DIR}" pull --ff-only; then :; else
        log "[warn] git pull 失败 (本地有改动或网络问题), 继续使用现有代码"
    fi
else
    log "git clone: ${REPO_URL} (分支 ${REPO_BRANCH}) -> ${REPO_DIR}"
    if ! git clone -b "${REPO_BRANCH}" "${clone_url}" "${REPO_DIR}"; then
        log "[error] git clone 失败 (私有仓请设置 GIT_TOKEN, 或用 REPO_URL 指定内部镜像)"
        exit 1
    fi
fi

# 测试套件目录: 独立仓 = 仓库根; 大仓嵌套布局 = 仓库根/$(SUITE_REL_PATH)
SUITE_DIR="${REPO_DIR}"
if [ -n "${SUITE_REL_PATH}" ]; then
    SUITE_DIR="${REPO_DIR}/${SUITE_REL_PATH}"
fi
if [ ! -d "${SUITE_DIR}" ]; then
    log "[error] 测试套件目录不存在: ${SUITE_DIR} (SUITE_REL_PATH=${SUITE_REL_PATH})"
    exit 1
fi
cd "${SUITE_DIR}" || exit 1
log "测试套件目录: ${SUITE_DIR}"
log "当前代码: $(git rev-parse --short HEAD) | $(git log -1 --pretty=%s | head -c 60)"

# ---------------------------------------------------------------- 2. 用例清单
# 优先级: 绝对路径 > 脚本目录外部清单 (用户自维护) > 仓库内置全量清单
if [[ "${TESTCASES_FILE}" == /* ]]; then
    CASES_FILE="${TESTCASES_FILE}"
    log "使用指定用例清单: ${CASES_FILE}"
elif [ -f "${SCRIPT_DIR}/${TESTCASES_FILE}" ]; then
    CASES_FILE="${SCRIPT_DIR}/${TESTCASES_FILE}"
    log "使用外部用例清单 (可自由增删用例, 不受 git 更新影响): ${CASES_FILE}"
else
    CASES_FILE="${SUITE_DIR}/${TESTCASES_FILE}"
    if [ ! -f "${CASES_FILE}" ]; then
        log "套件内无内置清单, 自动收集生成: ${CASES_FILE}"
        "${PYTHON}" tools/generate_testcases.py -o "${CASES_FILE}" \
            || { log "[error] 无法获得用例清单, 退出"; exit 1; }
    fi
    # 首次运行: 在脚本目录创建可编辑副本 (后续增删/注释 '#'-开头行即生效)
    cp -f "${CASES_FILE}" "${SCRIPT_DIR}/${TESTCASES_FILE}"
    CASES_FILE="${SCRIPT_DIR}/${TESTCASES_FILE}"
    log "已在脚本目录创建可编辑用例清单副本: ${CASES_FILE}"
    log "(后续可自由增删; 不想执行某条用例时整行注释或删除; 想重置可删除该文件)"
fi

if [ "${REGENERATE_CASES}" = "1" ]; then
    log "REGENERATE_CASES=1: 重新收集全量用例到套件内置清单..."
    "${PYTHON}" tools/generate_testcases.py -o "${SUITE_DIR}/${TESTCASES_FILE}" \
        || log "[warn] 全量清单重新生成失败"
    if [ -f "${SCRIPT_DIR}/${TESTCASES_FILE}" ]; then
        log "把新增用例同步进外部清单 (保留已有增删与注释)..."
        "${PYTHON}" tools/generate_testcases.py --sync "${SCRIPT_DIR}/${TESTCASES_FILE}" \
            || log "[warn] 外部清单同步失败"
    fi
fi

mapfile -t CASES < <(sed -e 's/\r$//' "${CASES_FILE}" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$')
TOTAL=${#CASES[@]}
if [ "${TOTAL}" -eq 0 ]; then
    log "[error] 用例清单为空 (全部被注释或删除), 退出"
    exit 1
fi
log "待执行用例: ${TOTAL} 条 (清单: ${CASES_FILE})"
log "日志目录:   ${LOG_DIR}"

# 无 timeout 命令时的兜底 (老系统)
TIMEOUT_BIN="timeout"
if ! command -v "${TIMEOUT_BIN}" > /dev/null 2>&1; then
    TIMEOUT_BIN=""
    log "[warn] 系统无 timeout 命令, 本次运行不设单用例超时"
fi

# ---------------------------------------------------------------- 3. 逐条执行
# 中断机制: 用例在后台执行 + wait。Ctrl+C 时 bash 的 wait 立即被打断并触发 trap,
# 对当前用例的整个进程组做 INT -> TERM -> KILL 升级终止 (pytest 会吞第一次 SIGINT
# 等待当前测试结束, 这是之前 "Ctrl+C 无法终止" 的根因), 然后汇总已完成用例后退出。
_active_child=""
_interrupted=0

_kill_child_tree() {
    # 终止当前用例进程组: timeout 自身是新进程组组长, 负号表示整组
    # (涵盖 pytest 及其派生的编译器/算子子进程); 组杀失败时退化为杀单进程。
    [ -z "${_active_child}" ] && return 0
    kill -INT -- "-${_active_child}" 2>/dev/null || kill -INT "${_active_child}" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-${_active_child}" 2>/dev/null || kill -TERM "${_active_child}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${_active_child}" 2>/dev/null || kill -KILL "${_active_child}" 2>/dev/null || true
}

on_interrupt() {
    if [ "${_interrupted}" -eq 1 ]; then
        log "[interrupt] 收到第二次中断, 立即强制退出 (已完成用例不再汇总)"
        exit 130
    fi
    _interrupted=1
    log "[interrupt] 收到 Ctrl+C, 正在终止当前用例 (INT->TERM->KILL)..."
    _kill_child_tree
    log "[interrupt] 当前用例已终止; 稍后汇总已完成用例 (再按一次 Ctrl+C 立即退出)"
}

trap on_interrupt INT TERM

printf 'order\tnodeid\texit_code\tseconds\tlog\n' > "${RAW_TSV}"
idx=0
for case_id in "${CASES[@]}"; do
    if [ "${_interrupted}" -eq 1 ]; then
        break
    fi
    idx=$((idx + 1))
    safe_name="$(printf '%s' "${case_id}" | sed -e 's#[/\\: ]#_#g' -e 's#\[#_#g' -e 's#\]#_#g')"
    log_file="${LOG_DIR}/$(printf '%03d' "${idx}")_${safe_name}.log"

    log "[${idx}/${TOTAL}] RUN ${case_id}"
    start_ts=$(date +%s)
    if [ -n "${TIMEOUT_BIN}" ]; then
        "${TIMEOUT_BIN}" "${CASE_TIMEOUT}" "${PYTHON}" -m pytest -v -s "${case_id}" ${EXTRA_PYTEST_ARGS} \
            > "${log_file}" 2>&1 &
    else
        "${PYTHON}" -m pytest -v -s "${case_id}" ${EXTRA_PYTEST_ARGS} \
            > "${log_file}" 2>&1 &
    fi
    _active_child=$!
    # 轮询等待而非阻塞 wait: sleep 结束时 shell 会处理待决信号并触发 trap
    # (阻塞 wait 在 msys/git-bash 下无法被 SIGINT 打断; Linux 上两者皆可)
    while kill -0 "${_active_child}" 2>/dev/null; do
        sleep 1
    done
    wait "${_active_child}"
    rc=$?
    _active_child=""
    if [ "${_interrupted}" -eq 1 ]; then
        rc=130
    fi
    end_ts=$(date +%s)
    dur=$((end_ts - start_ts))

    # 进度行状态 (权威解析在汇总脚本): 1 passed / 1 failed / 1 skipped / ...
    brief="$(grep -oE '[0-9]+ (passed|failed|skipped|error|errors)' "${log_file}" | tail -n 1)"
    if [ -z "${brief}" ]; then
        if [ "${rc}" -eq 124 ]; then brief="timeout"; fi
        if [ "${rc}" -eq 130 ]; then brief="interrupted"; fi
        [ -z "${brief}" ] && brief="rc=${rc}"
    fi
    log "[${idx}/${TOTAL}] ${brief} | 耗时 ${dur}s | 日志 ${log_file}"

    printf '%s\t%s\t%s\t%s\t%s\n' "${idx}" "${case_id}" "${rc}" "${dur}" "${log_file}" >> "${RAW_TSV}"
done

if [ "${_interrupted}" -eq 1 ]; then
    log "[interrupt] 执行被中止: 已完成 ${idx} 条用例, 汇总如下"
fi

# ---------------------------------------------------------------- 4. 汇总
if [ "${_interrupted}" -eq 1 ]; then
    log "汇总已完成用例 (执行被 Ctrl+C 中断)..."
else
    log "全部用例执行完毕, 汇总结果..."
fi
"${PYTHON}" tools/summarize_results.py --tsv "${RAW_TSV}" --out-dir "${RESULT_DIR}"
summary_rc=$?

echo
log "==================== 执行结束 ===================="
log "结果表格 : ${RESULT_DIR}/summary.csv / summary.md"
log "原始记录 : ${RAW_TSV}"
log "逐用例日志: ${LOG_DIR}/"
if [ "${_interrupted}" -eq 1 ]; then
    log "结论     : 执行被 Ctrl+C 中断, 以上为已完成用例的结果"
    exit 130
elif [ "${summary_rc}" -eq 0 ]; then
    log "结论     : 全部用例通过 (含跳过)"
else
    log "结论     : 存在失败/错误用例, 详见汇总报告"
fi

exit "${summary_rc}"
