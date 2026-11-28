# -*- coding: utf-8 -*-
"""实验 07：四臂验收（D 进卷，逐条 dev 存盘，补 P3 / P4）。

四臂：
  A   原样        : 只给中性 system
  C   只开风格药  : adapt(user_style only, max_tics=0)——实验 06 的老行为
  T   两味药      : adapt(user_style 2 条 + model_tics 2 条)
  D   随机扰动    : 对 A 的输出做同等幅度的随机结构扰动（本地生成，不调 API）

与 06 的差别：
  · D 臂进盲测卷（20 组 T vs A + 20 组 T vs D），P3 才可算
  · 逐条存每臂的"单条 dev"（4 个可单条算的指标），P4 才可算
  · C 与 T 同跑，看第二味药到底加了什么

用法：
  set AB_ASTRBOT_CONFIG=<AstrBot cmd_config.json>
  python experiments/run_07.py --n 120 --workers 12
产物：data/val7/{a,c,t,d}.jsonl、summary.json、sheet.md、key.json、per_item.json
"""

import argparse
import json
import os
import random
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from aiaccent.metrics import measure  # noqa: E402
from aiaccent.baseline import load, compare  # noqa: E402
from aiaccent.adapt import advise  # noqa: E402
from run_06 import (SYS_A, PARAMS, HOLDOUT, api_key, chat, chunks_of,  # noqa: E402
                    perturb, PROMPTS, USER_TURNS)

OUT = os.path.join(ROOT, "data", "val7")

# 单条就能算的指标（P4 用；其余指标在单条上无定义或噪声过大）
PER_ITEM = ["chunk_len_p50", "chunk_len_cv", "end_punct_rate", "hedge_per_kchar"]


def build_systems():
    user_turns = [json.loads(l) for l in open(USER_TURNS, encoding="utf-8")]
    only_style = advise(user_turns, max_user=2, max_tics=0)
    both = advise(user_turns, max_user=2, max_tics=2)
    return {
        "c": SYS_A + "\n" + "\n".join(only_style["instructions"]),
        "t": SYS_A + "\n" + "\n".join(both["instructions"]),
    }, only_style, both


def per_item_dev(turns, base):
    """每条 turn 的 |dev| 和（4 个可单条计算的指标）。"""
    out = []
    for t in turns:
        rep = measure([t])
        rows = compare(rep, base, experimental=True)
        d = {r["metric"]: r["dev"] for r in rows}
        out.append(round(sum(abs(d[m]) for m in PER_ITEM if m in d), 3))
    return out


def build_sheet(turns, base, n_pairs=20, seed=11):
    """20 组 T vs A + 20 组 T vs D，左右随机；key 带 pair 标记与逐条 dev。"""
    rng = random.Random(seed)
    n = min(len(turns["t"]), len(turns["a"]), len(turns["d"]))
    idx = list(range(n))
    rng.shuffle(idx)
    plan = []
    for k, i in enumerate(idx[:n_pairs]):
        plan.append((i, "a"))
    for k, i in enumerate(idx[n_pairs:2 * n_pairs]):
        plan.append((i, "d"))

    dev_a = per_item_dev(turns["a"], base)
    dev_t = per_item_dev(turns["t"], base)
    dev_d = per_item_dev(turns["d"], base)

    lines = ["# 盲测卷（实验 07：%d 组）" % len(plan), "",
             "每组两问：q1 更想接着聊哪个；q2 哪个读着更顺。",
             "在被选中的那段后面写 `Q1` / `Q2`；不想选就留空并写原因。", "", "---", ""]
    key = {}
    for g, (i, other) in enumerate(plan, start=1):
        t_text = turns["t"][i]
        o_text = turns[other][i]
        t_side = rng.choice(["1", "2"])
        o_side = "2" if t_side == "1" else "1"
        first, second = (t_text, o_text) if t_side == "1" else (o_text, t_text)
        lines.append("## %d" % g)
        lines.append("")
        lines.append("**1**")
        lines.append("")
        lines.extend(first)
        lines.append("")
        lines.append("**2**")
        lines.append("")
        lines.extend(second)
        lines.append("")
        lines.append("q1( ) q2( ) q3( )")
        lines.append("")
        lines.append("---")
        lines.append("")
        key[str(g)] = {"prompt_index": i, "other_arm": other,
                       "t_side": t_side, "dev_t": dev_t[i],
                       "dev_other": dev_a[i] if other == "a" else dev_d[i]}
    return "\n".join(lines), key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=20)
    args = ap.parse_args()

    prompts = [json.loads(l)["p"] for l in open(PROMPTS, encoding="utf-8")][:args.n]
    SYS, only_style, both = build_systems()
    print("C 臂（只风格）%d 条 / T 臂（两味药）%d 条"
          % (len(only_style["instructions"]), len(both["instructions"])))
    for x in both["instructions"]:
        print("  -", x)

    key_api = api_key()
    os.makedirs(OUT, exist_ok=True)
    systems = {"a": SYS_A, "c": SYS["c"], "t": SYS["t"]}

    jobs = [(g, i) for g in "act" for i in range(len(prompts))]
    results, lock, done, t0 = {}, threading.Lock(), [0], time.time()

    def work(job):
        g, i = job
        msg, usage = chat(key_api, systems[g], prompts[i])
        with lock:
            results[job] = (msg, usage)
            done[0] += 1
            if done[0] % 60 == 0:
                print("  %d/%d %.0fs" % (done[0], len(jobs), time.time() - t0), flush=True)

    threads = []
    for job in jobs:
        while len([t for t in threads if t.is_alive()]) >= args.workers:
            time.sleep(0.05)
        t = threading.Thread(target=work, args=(job,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    turns = {g: [] for g in "actd"}
    raw, ptok, ctok = {}, 0, 0
    for (g, i) in sorted(results):
        msg, usage = results[(g, i)]
        raw["%s:%d" % (g, i)] = {"prompt": prompts[i], "reply": msg, "usage": usage}
        if not msg:
            continue
        turns[g].append(chunks_of(msg))
        ptok += usage.get("prompt_tokens", 0)
        ctok += usage.get("completion_tokens", 0)

    for i in range(len(turns["a"])):
        rng = random.Random(7000 + i)
        turns["d"].append([x for x in perturb("\n".join(turns["a"][i]), rng).split("\n") if x])

    for g in "actd":
        with open(os.path.join(OUT, "%s.jsonl" % g), "w", encoding="utf-8") as f:
            for t in turns[g]:
                f.write(json.dumps([x for x in t if x], ensure_ascii=False) + "\n")
    json.dump(raw, open(os.path.join(OUT, "raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    base = load("chat")
    summary = {"n_prompts": len(prompts), "prompt_tokens": ptok,
               "completion_tokens": ctok,
               "c_instructions": only_style["instructions"],
               "t_instructions": both["instructions"], "arms": {}}
    for g in "actd":
        rep = measure(turns[g])
        rows = compare(rep, base, experimental=True)
        d = {r["metric"]: {"value": r["value"], "pct": r["pct"], "dev": r["dev"],
                           "verdict": r["verdict"]} for r in rows}
        hold = sum(abs(d[m]["dev"]) for m in HOLDOUT if m in d)
        summary["arms"][g] = {"n": len(turns[g]), "report": rep, "rows": d,
                              "holdout_abs_dev": round(hold, 3)}
        print("\n== %s (%d turns) 留出项 |dev| 之和 %.2f ==" % (g.upper(), len(turns[g]), hold))
        for r in rows:
            print("  %-22s %8s pct%5.1f dev%+7.2f %s" %
                  (r["metric"], round(r["value"], 3), r["pct"], r["dev"], r["verdict"]))

    sheet, key = build_sheet(turns, base, n_pairs=args.pairs)
    open(os.path.join(OUT, "sheet.md"), "w", encoding="utf-8").write(sheet)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    per_item = {"a": per_item_dev(turns["a"], base),
                "t": per_item_dev(turns["t"], base),
                "d": per_item_dev(turns["d"], base)}
    json.dump(per_item, open(os.path.join(OUT, "per_item.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    a_h = summary["arms"]["a"]["holdout_abs_dev"]
    t_h = summary["arms"]["t"]["holdout_abs_dev"]
    print("\nP2 泛化：T 留出 %.2f / A 留出 %.2f = %.2f（判据 ≤0.70）" %
          (t_h, a_h, t_h / max(a_h, 1e-9)))
    print("卷子：%s（%d 组）" % (os.path.join(OUT, "sheet.md"), len(key)))
    print("计分：python experiments/score_7_marks.py data/val7/marked.md "
          "--key data/val7/key.json")


if __name__ == "__main__":
    main()
