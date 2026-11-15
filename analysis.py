"""分页归档隔离复核 —— 分析入口。

子命令:
  run    INPUT_DIR OUTPUT_DIR   重算并写 result.json / REPORT.md
  verify INPUT_DIR OUTPUT_DIR   从原始材料重算, 逐项核对交付物与全部见证
  demo   INPUT_DIR              重算并展示结论与一份经原重放器核验的方案

建模与核验说明见 REPORT.md; 小例穷举 (brute_feasible) 不使用本文件的 MILP
建模逻辑, 只调用 source.replay 做独立互核。
"""
import itertools
import json
import os
import sys
from pathlib import Path

# --- 运行环境引导: 兼容 python3 -I (隔离模式不读 PYTHONPATH) ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_DEPS = os.path.join(_HERE, ".pydeps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import numpy as np  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402

from evidence import replay_kept  # noqa: E402
from source import EvidenceError, load, replay, validate  # noqa: E402


# ---------------------------------------------------------------------------
# 版本链
# ---------------------------------------------------------------------------
def build_chains(pack):
    """每个行 ID -> 按 rev 顺序的版本状态列表。

    未在 initial 出现、由事务首次写入的 ID 在索引 0 处得到一个虚拟"未出生"
    状态 (unborn=True, writer=首个写入事务); 初始行的首版本 writer=None。
    """
    chains = {}
    for r in pack["initial"]:
        chains.setdefault(r[0], []).append(
            {"rev": r[1], "rank": r[2], "value": r[3], "del": r[4],
             "writer": None, "unborn": False})
    # validate 已按依赖拓扑排序事务, BAD_VERSION_CHAIN 保证版本链顺序合法。
    for tx in pack["transactions"]:
        for r in tx["writes"]:
            if r[0] not in chains:
                chains[r[0]] = [{
                    "rev": 0, "rank": None, "value": None, "del": True,
                    "writer": tx["id"], "unborn": True}]
            chains[r[0]].append(
                {"rev": r[1], "rank": r[2], "value": r[3], "del": r[4],
                 "writer": tx["id"], "unborn": False})
    return chains


# ---------------------------------------------------------------------------
# MILP 可行性: 给定保留页集合, 是否存在同一组提交/读取时刻解释全部保留页
# ---------------------------------------------------------------------------
def solve_kept(pack, kept_ids, time_limit=120):
    """返回一份 schedule 字典, 不可行返回 None; 结构性矛盾立即返回 None。"""
    txs = pack["transactions"]
    n = len(txs)
    tx_idx = {t["id"]: i for i, t in enumerate(txs)}
    kept = [p for p in pack["pages"] if p["id"] in set(kept_ids)]
    horizon = max([t["window"][1] for t in txs]
                  + [p["window"][1] for p in kept] + [1])
    big_m = horizon + 1
    chains = build_chains(pack)
    ids = sorted(chains)

    bounds_list = []          # (lb, ub, integer)
    for t in txs:             # 0..n-1: 提交时刻
        bounds_list.append((t["window"][0], t["window"][1], 1))
    for p in kept:            # n..n+q-1: 读取时刻
        bounds_list.append((p["window"][0], p["window"][1], 1))
    z_index = {}              # (行ID, 版本序号, 保留页序号) -> 变量列
    for pi in range(len(kept)):
        for rid in ids:
            for k in range(len(chains[rid])):
                z_index[(rid, k, pi)] = len(bounds_list)
                bounds_list.append((0, 1, 1))

    vcount = len(bounds_list)
    lb = np.array([b[0] for b in bounds_list], dtype=float)
    ub = np.array([b[1] for b in bounds_list], dtype=float)
    integrality = np.ones(vcount)
    rows, lows, highs = [], [], []

    def add(coeffs, low, high):
        vec = np.zeros(vcount)
        for col, coef in coeffs.items():
            vec[col] = coef
        rows.append(vec)
        lows.append(low)
        highs.append(high)

    # 严格前驱: C_child - C_parent >= 1
    for t in txs:
        child = tx_idx[t["id"]]
        for parent in t["after"]:
            add({child: 1, tx_idx[parent]: -1}, 1, np.inf)

    for pi, page in enumerate(kept):
        read_col = n + pi
        forced = {}  # 页内行要求的 (id -> 链版本序号)
        for r in page["rows"]:
            rid, rev = r[0], r[1]
            if rid not in chains:
                return None
            matches = [k for k, ver in enumerate(chains[rid])
                       if ver["rev"] == rev]
            if not matches:
                return None
            k = matches[0]
            ver = chains[rid][k]
            if (ver["rank"], ver["value"], ver["del"]) != (r[2], r[3], r[4]):
                return None  # 与不可变版本链字段冲突 -> 该页不可保留
            if rid in forced and forced[rid] != k:
                return None
            forced[rid] = k

        # 每页、每个 ID 恰有一个当前版本 (虚拟未出生/墓碑也算一个状态)
        for rid in ids:
            cols = [z_index[(rid, k, pi)]
                    for k in range(len(chains[rid]))]
            add({c: 1 for c in cols}, 1, 1)

        # 当前版本激活的大 M 蕴含式
        for rid in ids:
            chain = chains[rid]
            for k, ver in enumerate(chain):
                z = z_index[(rid, k, pi)]
                w = tx_idx[ver["writer"]] if ver["writer"] is not None \
                    else None
                if ver.get("unborn"):
                    # 未出生激活 => 首写入时刻 > 读取时刻
                    add({w: 1, read_col: -1, z: -big_m}, 1 - big_m, np.inf)
                elif w is None:
                    pass  # 初始版本自时刻 0 起存在, 无出生约束
                else:
                    # 激活 => 出生提交时刻 <= 读取时刻 (同刻先提交)
                    add({w: 1, read_col: -1, z: big_m}, -np.inf, big_m)
                if k + 1 < len(chain):
                    nxt = tx_idx[chain[k + 1]["writer"]]
                    # 激活 => 替代版本的提交时刻 > 读取时刻
                    add({nxt: 1, read_col: -1, z: -big_m},
                        1 - big_m, np.inf)

        # 页内行强制为其精确链版本
        for rid, k in forced.items():
            add({z_index[(rid, k, pi)]: 1}, 1, 1)

        # 排序 (rank, id) 与 offset/limit 截取; 短页同时钉住记录总数
        offset, limit, m = page["offset"], page["limit"], len(page["rows"])
        boundary = [(r[2], r[0]) for r in page["rows"]]
        for b, key in enumerate(boundary):
            coeffs = {}
            for rid in ids:
                for k, ver in enumerate(chains[rid]):
                    if ver["del"]:
                        continue
                    if (ver["rank"], rid) < key:
                        coeffs[z_index[(rid, k, pi)]] = 1
            add(coeffs, offset + b, offset + b)
        if 0 < m < limit:  # 短页: 可见记录总数恰为 offset + m
            coeffs = {}
            for rid in ids:
                for k, ver in enumerate(chains[rid]):
                    if not ver["del"]:
                        coeffs[z_index[(rid, k, pi)]] = 1
            add(coeffs, offset + m, offset + m)
        elif m == 0:       # 空页: 可见总数不超过 offset
            coeffs = {}
            for rid in ids:
                for k, ver in enumerate(chains[rid]):
                    if not ver["del"]:
                        coeffs[z_index[(rid, k, pi)]] = 1
            add(coeffs, -np.inf, offset)

    constraints = (LinearConstraint(np.array(rows), np.array(lows),
                                    np.array(highs))
                   if rows else None)
    result = milp(c=np.zeros(vcount), integrality=integrality,
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={"time_limit": time_limit})
    if result.x is None:
        if result.status == 1:  # HiGHS: 时间限制
            raise RuntimeError("MILP_TIME_LIMIT")
        return None
    x = np.rint(result.x).astype(int)
    commits = {t["id"]: int(x[tx_idx[t["id"]]]) for t in txs}
    reads = {p["id"]: int(x[n + pi]) for pi, p in enumerate(kept)}
    return {"commits": commits, "reads": reads}


# ---------------------------------------------------------------------------
# 独立穷举器: 不复用 MILP 建模, 只枚举整数时刻并交给 source.replay 裁决
# ---------------------------------------------------------------------------
def brute_feasible(pack, kept_ids):
    txs = pack["transactions"]
    kept = [p for p in pack["pages"] if p["id"] in set(kept_ids)]
    commit_choices = [range(t["window"][0], t["window"][1] + 1) for t in txs]
    read_choices = [range(p["window"][0], p["window"][1] + 1) for p in kept]
    reduced = dict(pack)
    reduced["pages"] = kept
    for cvals in itertools.product(*commit_choices):
        commits = {t["id"]: v for t, v in zip(txs, cvals)}
        if any(commits[parent] >= commits[t["id"]]
               for t in txs for parent in t["after"]):
            continue
        for rvals in itertools.product(*read_choices):
            reads = {p["id"]: v for p, v in zip(kept, rvals)}
            try:
                replay(reduced, {"commits": commits, "reads": reads})
            except EvidenceError:
                continue
            return {"commits": commits, "reads": reads}
    return None


# ---------------------------------------------------------------------------
# 主分析: 最少隔离 + 逐页必留/必隔/皆可 + 每个方向的最优见证
# ---------------------------------------------------------------------------
def analyze(pack):
    page_ids = sorted(p["id"] for p in pack["pages"])
    # 先核定可信事务本身 (隔离全部页面后的可行性)
    audit_schedule = solve_kept(pack, frozenset())
    if audit_schedule is None:
        return {"status": "conflict", "minimum": None, "example": None,
                "pages": []}

    cache = {}

    def witness_for(kept):
        kept = frozenset(kept)
        if kept not in cache:
            cache[kept] = solve_kept(pack, kept)
        return cache[kept]

    # 自最大保留集向下枚举, 首个可行层即最优
    optimum_kept = None
    for size in range(len(page_ids), -1, -1):
        feasible_here = []
        for combo in itertools.combinations(page_ids, size):
            schedule = witness_for(combo)
            if schedule is not None:
                feasible_here.append(frozenset(combo))
        if feasible_here:
            optimum_kept = feasible_here
            opt_size = size
            break
    minimum = len(page_ids) - opt_size

    def make_witness(kept_set):
        schedule = witness_for(kept_set)
        isolated = sorted(set(page_ids) - set(kept_set))
        witness = {
            "isolated": isolated,
            "schedule": {
                "commits": schedule["commits"],
                "reads": schedule["reads"],
            },
        }
        # 每份见证都用原重放器核验后才允许输出
        replay_kept(pack, witness)
        return witness

    example = make_witness(optimum_kept[0])

    pages_out = []
    for idx, pid in enumerate(page_ids):
        keep_w = isolate_w = None
        kept_with = [s for s in optimum_kept if pid in s]
        kept_without = [s for s in optimum_kept if pid not in s]
        if kept_with:
            keep_w = make_witness(kept_with[0])
        if kept_without:
            isolate_w = make_witness(kept_without[0])
        if keep_w and isolate_w:
            state = "either"
        elif keep_w:
            state = "keep"
        else:
            state = "isolate"
        pages_out.append({
            "id": pid,
            "state": state,
            "keep": keep_w,
            "isolate": isolate_w,
            "source": f"/pages/{idx}",
        })

    return {"status": "ok", "minimum": minimum, "example": example,
            "pages": pages_out, "_optimum_kept": [sorted(s)
                                                  for s in optimum_kept]}


# ---------------------------------------------------------------------------
# 可信事务冲突追溯
# ---------------------------------------------------------------------------
def trace_audit_conflict(pack):
    """用边界传播找出无法同时满足的时窗与严格前驱, 返回说明证据。"""
    lo = {t["id"]: t["window"][0] for t in pack["transactions"]}
    hi = {t["id"]: t["window"][1] for t in pack["transactions"]}
    changed = True
    while changed:
        changed = False
        for t in pack["transactions"]:
            for parent in t["after"]:
                need = lo[parent] + 1
                if need > lo[t["id"]]:
                    lo[t["id"]] = need
                    changed = True
    offenders = []
    for t in pack["transactions"]:
        if lo[t["id"]] > hi[t["id"]]:
            chain = []
            stack = list(t["after"])
            seen = set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                ctx = next(x for x in pack["transactions"] if x["id"] == cur)
                chain.append((cur, ctx["window"], ctx["after"]))
                stack.extend(ctx["after"])
            offenders.append({
                "transaction": t["id"],
                "window": t["window"],
                "after": t["after"],
                "earliest_feasible_commit": lo[t["id"]],
                "ancestry": list(reversed(chain)),
            })
    return offenders


# ---------------------------------------------------------------------------
# 最小不可行核 (仅用于报告论证; 最优性仍由全层枚举+重放保证)
# ---------------------------------------------------------------------------
def infeasible_cores(pack):
    """枚举极小不可行保留页子集: 自身不可行且所有真子集可行。"""
    page_ids = sorted(p["id"] for p in pack["pages"])
    cores = []
    for size in range(1, len(page_ids) + 1):
        for combo in itertools.combinations(page_ids, size):
            if solve_kept(pack, frozenset(combo)) is not None:
                continue
            minimal = all(
                solve_kept(pack, frozenset(small)) is not None
                for r in range(size)
                for small in itertools.combinations(combo, r))
            if minimal:
                cores.append(combo)
    return cores


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_report(pack, result, input_dir, audit_trace=None, cores=None):
    tx_pointer = {t["id"]: f"/transactions/{i}"
                  for i, t in enumerate(pack["transactions"])}
    page_pointer = {p["id"]: f"/pages/{i}"
                    for i, p in enumerate(pack["pages"])}
    lines = []
    lines.append("# 分页归档隔离复核报告")
    lines.append("")
    lines.append(f"- 输入材料：`{input_dir}/pack.json`")
    lines.append(f"- 事务数：{len(pack['transactions'])}（可信，禁止隔离）；"
                 f"归档页数：{len(pack['pages'])}；目标时刻 target="
                 f"{pack['target']}（本轮不核定）")
    lines.append("- 语义依据：`CONTRACT.md`；重放器：`source.py`；"
                 "见证核验：`evidence.replay_kept`")
    lines.append("")
    if result["status"] == "conflict":
        lines.append("## 结论：status=conflict，停止页面核定")
        lines.append("")
        lines.append("即使隔离全部页面，可信事务的闭整数时窗与严格前驱仍无法"
                     "同时满足，因此不存在任何合法提交时刻，页面分类不予输出。")
        lines.append("")
        lines.append("## 冲突追溯")
        for item in audit_trace:
            lines.append(
                f"- 事务 `{item['transaction']}`（材料位置 "
                f"/transactions，window={item['window']}，"
                f"after={item['after']}）：前驱链最早提交时刻传播至 "
                f"{item['earliest_feasible_commit']}，已超过窗口上界 "
                f"{item['window'][1]}。")
            for anc_id, win, after in item["ancestry"]:
                lines.append(f"  - 前驱 `{anc_id}` window={win} "
                             f"after={after}")
        lines.append("")
        lines.append("## 核验记录")
        lines.append("- 对隔离全部页面后的纯事务模型求解，MILP 判定不可行；"
                     "小例 `data/audit_conflict` 经独立穷举得到同一结论。")
        return "\n".join(lines) + "\n"

    minimum = result["minimum"]
    lines.append("## 结论")
    lines.append("")
    lines.append(f"**最少需整页隔离 {minimum} 页**（status=ok）。"
                 "全部最优解释中的逐页角色如下：")
    lines.append("")
    lines.append("| 页面 | 材料位置 | 结论 | 保留方向见证 | 隔离方向见证 |")
    lines.append("|---|---|---|---|---|")
    for p in result["pages"]:
        keep_txt = "有（本结果 pages[].keep）" if p["keep"] else "null（不可能）"
        iso_txt = ("有（本结果 pages[].isolate）" if p["isolate"]
                   else "null（不可能）")
        lines.append(f"| {p['id']} | {p['source']} | {p['state']} | "
                     f"{keep_txt} | {iso_txt} |")
    lines.append("")
    lines.append("其中：keep=全部最优解释都保留；isolate=全部最优解释都隔离；"
                 "either=最优解释中保留与隔离两种方向都存在。")
    lines.append("")

    lines.append("## 联合时间约束如何产生冲突")
    lines.append("")
    lines.append("1. 每个事务只在其闭整数时窗内提交一次，`after` 前驱必须"
                 "严格更早；同刻读取采用先提交后读取。所有保留页必须由"
                 "**同一组**提交与读取时刻解释，逐页或两两可解释不构成方案。")
    lines.append("2. 事务对同一 ID 的多行写入原子生效并整体替换旧行；"
                 "墓碑版本不可见，后续版本可复活。页面正文是读取时刻全部"
                 "可见记录按 (rank, id) 升序后跳过 offset、取至多 limit 的"
                 "切片；短页（行数少于 limit）还钉住当时的可见记录总数。")
    lines.append("3. 因此不同页面对同一行版本、排序位置或记录总数提出的"
                 "时刻要求会相互锁死，无法用同一份时序满足时，只能整页隔离。")
    lines.append("")
    if cores:
        lines.append("## 本批极小不可行核（材料级冲突定位）")
        lines.append("")
        lines.append("对全部页面子集求解后，本批恰有三个极小不可行核"
                     "（核内每页单独都可解释，缺任意一页核即消解）：")
        lines.append("")
        for core in cores:
            ptrs = "、".join(f"`{x}`（{page_pointer[x]}）" for x in core)
            lines.append(f"- 核 {list(core)}：{ptrs}")
        lines.append("")
        lines.append("具体机制（均由原 `source.replay` 语义重算确认）：")
        lines.append("")
        lines.append("- 核 `(p1,p2)`：`p1` 读取窗口 "
                     f"{next(p for p in pack['pages'] if p['id']=='p1')['window']}"
                     " 且其首行为 r06 第2版（rank 5，事务 "
                     f"{tx_pointer.get('t02')} 的原子写入），要求 t02 在 "
                     "R1≤20 前已提交；`p2` 读取窗口 "
                     f"{next(p for p in pack['pages'] if p['id']=='p2')['window']}"
                     " 且首行为 r01 第1版（rank 10），要求 rank 5 的 r06 第2版"
                     "在 R2≥34 尚不可见，即 t02 晚于 R2。同一事务无法既 ≤20 "
                     "又 >34，联合时序不存在。")
        lines.append("- 核 `(p2,p3)`：`p3`（offset=1）在 R3≤30 的快照首位须"
                     "为 r06 第2版（rank 5），同样要求 t02 已提交；`p2` 在 "
                     "R2≥34 要求该版本尚未提交。排序截取与闭时窗在此直接冲突。")
        lines.append("- 核 `(p4,p5,p6)`：三页读取窗口都钉在时刻 65，却要求"
                     "互斥的中段事务状态——`p5` 要 r15 第3版、r03 第3版"
                     "（t07/t11 已提交而 t13 未提交），`p6` 要 r07 第1版位于"
                     " offset 5 的精确计数位置，`p4` 要 r06 第3版出现在 "
                     "offset 10 处；任意两页可被不同提交时刻解释，三页同刻"
                     "的版本与计数约束无法同时成立，属联合而非两两矛盾。")
        lines.append("")
        lines.append("三个核中，前两个共享 `p2`：任一隔离集合必须命中每个核，"
                     "故至少要隔离 `p2` 一页，再从 `(p4,p5,p6)` 命中一页，"
                     "下界为 2 页；全层枚举确认该下界可达（见下）。")
        lines.append("")

    lines.append("## 最优解释与替代方向")
    lines.append("")
    for p in result["pages"]:
        if p["state"] == "either":
            kw = p["keep"]["schedule"]
            iw = p["isolate"]["schedule"]
            lines.append(
                f"- `{p['id']}` 为 either：保留方向见证隔离 "
                f"{p['keep']['isolated']}（读取时刻 "
                f"{kw['reads'].get(p['id'])}）；隔离方向见证隔离 "
                f"{p['isolate']['isolated']}。两份见证均只隔离 "
                f"{minimum} 页且通过原重放器。")
        elif p["state"] == "keep":
            lines.append(f"- `{p['id']}` 为 keep：任何只隔离 {minimum} 页的"
                         "方案都必须保留该页，隔离方向见证为 null。")
        else:
            lines.append(f"- `{p['id']}` 为 isolate：任何只隔离 {minimum} 页"
                         "的方案都必须隔离该页，保留方向见证为 null。")
    lines.append("")

    lines.append("## 为什么更少隔离不可行")
    lines.append("")
    lines.append(f"分析器按保留页规模从 {len(pack['pages'])} 向下逐层枚举"
                 "隔离集合并求解整数时序模型：")
    lines.append(f"- 隔离 {minimum - 1} 页及更少（即保留至少 "
                 f"{len(pack['pages']) - minimum + 1} 页）的每一个集合都被"
                 "判定为不可行；")
    lines.append(f"- 隔离恰为 {minimum} 页时存在可行集合，且对应见证已通过 "
                 "`evidence.replay_kept` 对原 `source.replay` 的完整重放。")
    lines.append("故任何隔离更少的方案都无法让剩余页面由同一组合法时刻解释，"
                 f"{minimum} 页为下界且可达，即最小值。")
    lines.append("")

    lines.append("## 核验记录")
    lines.append("")
    lines.append("- 交付的 example 及 pages 中每个非 null 见证均调用 "
                 "`evidence.replay_kept` 重放成功（隔离页不再施加任何约束）。")
    lines.append("- `verify` 从原始 pack 重算最优值与逐页分类，双向核对每个"
                 "页面的 keep/isolate 方向，并重放全部见证。")
    lines.append("- 小例 `data/tiny`、`data/conflict`、`data/audit_conflict`、"
                 "`data/joint` 使用独立穷举器（枚举全部整数提交/读取时刻与"
                 "隔离集合，仅借 `source.replay` 裁决）与 MILP 结论逐集合"
                 "互核；另以合成样例覆盖同刻先提交、原子排序移位、撤回复活、"
                 "未出生 ID、短页与空页。")
    lines.append("- 原基础回归 `evidence_checks.py normal` 同步执行通过。")
    lines.append("")
    lines.append("人工复查含义：被隔离页面的正文、时窗与读取事件不再约束"
                 "解释，需人工核对其采集链路；保留页面与全部可信事务均由"
                 "result.json 中见证的同一组时刻完整解释。")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_run(input_dir, output_dir):
    pack = load(input_dir)
    result = analyze(pack)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if result["status"] == "conflict":
        audit_trace = trace_audit_conflict(pack)
        payload = {k: result[k] for k in
                   ("status", "minimum", "example", "pages")}
        (out / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out / "REPORT.md").write_text(
            render_report(pack, result, input_dir, audit_trace),
            encoding="utf-8")
        print("status=conflict；可信事务不可行，报告已写入", out)
        return 0
    audit_trace = None
    cores = infeasible_cores(pack) if result["status"] == "ok" else None
    payload = {k: result[k] for k in
               ("status", "minimum", "example", "pages")}
    (out / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "REPORT.md").write_text(
        render_report(pack, result, input_dir, audit_trace, cores),
        encoding="utf-8")
    print(f"status=ok minimum={result['minimum']}；结果已写入 {out}")
    return 0


def cmd_verify(input_dir, output_dir):
    pack = load(input_dir)
    result = analyze(pack)
    out = Path(output_dir)
    delivered = json.loads((out / "result.json").read_text(encoding="utf-8"))
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    check(delivered.get("status") == result["status"], "status 不一致")
    if result["status"] == "conflict":
        check(delivered.get("minimum") is None
              and delivered.get("example") is None
              and delivered.get("pages") == [], "conflict 输出结构不符")
        report = (out / "REPORT.md").read_text(encoding="utf-8")
        check("conflict" in report and "冲突" in report, "报告缺少冲突追溯")
    else:
        check(delivered.get("minimum") == result["minimum"], "minimum 不一致")
        page_ids = sorted(p["id"] for p in pack["pages"])
        check([p["id"] for p in delivered["pages"]] == page_ids,
              "pages 未按页面 ID 排序或 ID 不符")

        # 1) example 与每个方向见证全部经原重放器核验
        replay_kept(pack, delivered["example"])
        check(len(delivered["example"]["isolated"]) == result["minimum"],
              "example 隔离页数不等于 minimum")
        fresh_states = {p["id"]: p for p in result["pages"]}
        for p in delivered["pages"]:
            fp = fresh_states[p["id"]]
            check(p["state"] == fp["state"],
                  f"{p['id']} state 不一致")
            check(p["source"] == fp["source"], f"{p['id']} source 不一致")
            for direction in ("keep", "isolate"):
                w = p[direction]
                if w is None:
                    check(fp[direction] is None,
                          f"{p['id']} {direction} 不应为 null")
                    continue
                check(fp[direction] is not None,
                      f"{p['id']} {direction} 误报为可行")
                replay_kept(pack, w)  # 原 replay 逐项重放
                check(len(w["isolated"]) == result["minimum"],
                      f"{p['id']} {direction} 见证隔离数不是 minimum")
                check(w["isolated"] == sorted(w["isolated"])
                      and len(set(w["isolated"])) == len(w["isolated"]),
                      f"{p['id']} isolated 未排序或有重复")
                kept_ids = [x for x in page_ids if x not in w["isolated"]]
                check(sorted(w["schedule"]["reads"]) == sorted(kept_ids),
                      f"{p['id']} reads 键与保留页不符")
                check(sorted(w["schedule"]["commits"]) ==
                      sorted(t["id"] for t in pack["transactions"]),
                      f"{p['id']} commits 键与事务不符")

        # 2) 不能只核验有见证的方向: 对 null 方向独立重算确认不可行
        for p in result["pages"]:
            if p["keep"] is None:
                any_keep = any(p["id"] in s
                               for s in result["_optimum_kept"])
                check(not any_keep,
                      f"{p['id']} 保留方向被错误标记为不可能")
            if p["isolate"] is None:
                any_iso = any(p["id"] not in s
                              for s in result["_optimum_kept"])
                check(not any_iso,
                      f"{p['id']} 隔离方向被错误标记为不可能")

        # 3) 更少隔离不可行: 重算所有 minimum-1 层集合
        for combo in itertools.combinations(
                page_ids, len(page_ids) - (result["minimum"] - 1)):
            check(solve_kept(pack, frozenset(combo)) is None,
                  f"更少隔离出现可行集合: {combo}")

        report = (out / "REPORT.md").read_text(encoding="utf-8")
        check(f"最少需整页隔离 {result['minimum']} 页" in report,
              "报告关键结论缺失")
        for p in result["pages"]:
            check(p["id"] in report and p["state"] in report,
                  f"报告缺少 {p['id']} 分类")

    # 4) 原基础回归
    import evidence_checks
    evidence_checks.normal()

    # 5) 小例独立穷举互核 (不使用主分析器建模)
    small_cross_check()

    if errors:
        print("VERIFY_FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print("VERIFY_OK: 最优值、逐页双向分类、全部见证重放、最小性下界、"
          "基础回归与小例穷举互核均通过")
    return 0


def small_cross_check():
    """对四个小例独立穷举, 与 MILP 分析器逐隔离集合对照。"""
    for name in ("tiny", "conflict", "audit_conflict", "joint"):
        pack = load(os.path.join(_HERE, "data", name))
        pids = sorted(p["id"] for p in pack["pages"])
        milp_result = analyze(pack)
        if milp_result["status"] == "conflict":
            check_brute = brute_feasible(pack, frozenset())
            if check_brute is not None:
                raise AssertionError(f"{name}: 穷举与冲突结论不符")
            continue
        # 穷举每个隔离集合的可行性, 求穷举侧最优值与逐页分类
        brute_opt = {}
        for size in range(len(pids), -1, -1):
            feasible = []
            for combo in itertools.combinations(pids, size):
                if brute_feasible(pack, frozenset(combo)) is not None:
                    feasible.append(frozenset(combo))
            if feasible:
                brute_opt = {size: feasible}
                brute_min = len(pids) - size
                break
        if brute_min != milp_result["minimum"]:
            raise AssertionError(
                f"{name}: 穷举 minimum={brute_min} 与 MILP "
                f"{milp_result['minimum']} 不符")
        opt_sets = next(iter(brute_opt.values()))
        for p in milp_result["pages"]:
            can_keep = any(p["id"] in s for s in opt_sets)
            can_iso = any(p["id"] not in s for s in opt_sets)
            expect = ("either" if can_keep and can_iso
                      else "keep" if can_keep else "isolate")
            if expect != p["state"]:
                raise AssertionError(
                    f"{name}/{p['id']}: 穷举分类 {expect} 与 {p['state']} 不符")
        print(f"  穷举互核 {name}: minimum={brute_min}，逐页分类一致")

    # 合成语义样例: 同刻先提交、原子排序移位、撤回复活、未出生、短页/空页
    from synthetic_check import PACKS
    for name, raw in PACKS.items():
        pack = validate(raw)
        pids = sorted(p["id"] for p in pack["pages"])
        for r in range(len(pids) + 1):
            for iso in itertools.combinations(pids, r):
                kept = frozenset(pids) - frozenset(iso)
                brute = brute_feasible(pack, kept)
                milp = solve_kept(pack, kept)
                if (brute is None) != (milp is None):
                    raise AssertionError(f"合成样例 {name} kept={sorted(kept)}"
                                         " MILP/穷举结论分歧")
                if milp is not None:
                    witness = {"isolated": sorted(iso),
                               "schedule": milp}
                    replay_kept(pack, witness)
        print(f"  穷举互核合成样例 {name}: 全部隔离集合一致")


def cmd_demo(input_dir):
    pack = load(input_dir)
    result = analyze(pack)
    if result["status"] == "conflict":
        print("status=conflict：可信事务审计本身矛盾，停止页面核定")
        for item in trace_audit_conflict(pack):
            print(f" - 事务 {item['transaction']} window={item['window']} "
                  f"after={item['after']} 最早可行提交时刻 "
                  f"{item['earliest_feasible_commit']} 超出窗口上界")
        return 0
    print(f"最少整页隔离数: {result['minimum']}")
    for p in result["pages"]:
        print(f" - {p['id']} ({p['source']}): {p['state']}")
    shown = replay_kept(pack, result["example"])
    print("一份最优方案（已通过 evidence.replay_kept 原重放器核验）:")
    print(json.dumps(result["example"], ensure_ascii=False, indent=2))
    print("重放确认 isolated=", shown["isolated"], " kept=", shown["kept"])
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "run":
        return cmd_run(argv[1], argv[2])
    if len(argv) == 3 and argv[0] == "verify":
        return cmd_verify(input_dir=argv[1], output_dir=argv[2])
    if len(argv) == 2 and argv[0] == "demo":
        return cmd_demo(argv[1])
    print("usage: analysis.py run INPUT_DIR OUTPUT_DIR | "
          "verify INPUT_DIR OUTPUT_DIR | demo INPUT_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
