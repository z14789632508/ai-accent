# -*- coding: utf-8 -*-
"""实验 08：第三味药（交往规则）有没有用。

两臂：
  T  两味药（风格 + 毛病）—— 实验 07 的胜者
  R  三味药（再加 2 条交往规则：熟人该损 / 被夸要回敬）

这轮**不做盲测卷**（人一天填两次会疲劳），只做两件事：
  1. 机器侧：两臂的分布指标差多少（预期：结构指标基本不动，动的是内容）
  2. 人侧抽样：把同 prompt 的两段并排打出来，看"有没有损/有没有回敬"

用法：python experiments/run_08.py --n 40
"""

import argparse
import json
import os
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
from run_06 import SYS_A, PARAMS, api_key, chat, chunks_of, PROMPTS, USER_TURNS  # noqa: E402

OUT = os.path.join(ROOT, "data", "val8")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    user_turns = [json.loads(l) for l in open(USER_TURNS, encoding="utf-8")]
    t = advise(user_turns, max_user=2, max_tics=2, max_rel=0)
    r = advise(user_turns, max_user=2, max_tics=2, max_rel=2)
    print("T 臂 %d 条 / R 臂 %d 条" % (len(t["instructions"]), len(r["instructions"])))
    for x in r["instructions"][len(t["instructions"]):]:
        print("  新增:", x)

    systems = {"t": SYS_A + "\n" + "\n".join(t["instructions"]),
               "r": SYS_A + "\n" + "\n".join(r["instructions"])}
    prompts = [json.loads(l)["p"] for l in open(PROMPTS, encoding="utf-8")][:args.n]

    key = api_key()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(g, i) for g in "tr" for i in range(len(prompts))]
    results, lock, done, t0 = {}, threading.Lock(), [0], time.time()

    def work(job):
        g, i = job
        msg, usage = chat(key, systems[g], prompts[i])
        with lock:
            results[job] = (msg, usage)
            done[0] += 1
            if done[0] % 20 == 0:
                print("  %d/%d %.0fs" % (done[0], len(jobs), time.time() - t0), flush=True)

    threads = []
    for job in jobs:
        while len([x for x in threads if x.is_alive()]) >= args.workers:
            time.sleep(0.05)
        th = threading.Thread(target=work, args=(job,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    turns = {"t": [], "r": []}
    pairs = []
    for i in range(len(prompts)):
        a = results.get(("t", i), ("", {}))[0]
        b = results.get(("r", i), ("", {}))[0]
        if a:
            turns["t"].append(chunks_of(a))
        if b:
            turns["r"].append(chunks_of(b))
        pairs.append({"prompt": prompts[i], "T": a, "R": b})

    json.dump(pairs, open(os.path.join(OUT, "pairs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for g in "tr":
        with open(os.path.join(OUT, "%s.jsonl" % g), "w", encoding="utf-8") as f:
            for x in turns[g]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    base = load("chat")
    for g in "tr":
        rep = measure(turns[g])
        rows = compare(rep, base, experimental=True)
        print("\n== %s (%d turns) ==" % (g.upper(), len(turns[g])))
        for row in rows:
            print("  %-22s %8s pct%5.1f dev%+7.2f %s" %
                  (row["metric"], round(row["value"], 3), row["pct"], row["dev"],
                   row["verdict"]))

    print("\n== 抽样 %d 组（T 两味药 / R 三味药）==" % args.show)
    for k, p in enumerate(pairs[:args.show], start=1):
        print("\n[%d] %s" % (k, p["prompt"]))
        print("  T>", p["T"].replace("\n", " ⏎ "))
        print("  R>", p["R"].replace("\n", " ⏎ "))


if __name__ == "__main__":
    main()
