# -*- coding: utf-8 -*-
"""实验 07 的计分：T 臂 vs A 臂（P1）、T 臂 vs D 臂（P3）、逐条 dev 与人选择的相关（P4）。

卷面标记与 06 相同：在选中那段后面写 `Q1` / `Q2`，留空表示弃权。

用法：
    python experiments/score_7_marks.py data/val7/marked.md --key data/val7/key.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from score_6_marks import parse, choice_for, binom_two_sided


def spearman(xs, ys):
    """秩相关；有并列时用平均秩。"""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--key", default="data/val7/key.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    key = {str(k): v for k, v in json.load(open(args.key, encoding="utf-8")).items()}
    groups, notes = parse(args.sheet)

    rows = []
    for g in sorted(groups, key=lambda s: int(s)):
        if g not in key:
            continue
        k = key[g]
        t_side = int(k["t_side"])
        o_side = 3 - t_side
        picks = {}
        for q in (1, 2):
            s = choice_for(groups[g], q)
            picks[q] = None if s is None else ("T" if s == t_side else "O")
        rows.append({"group": int(g), "pair": k["other_arm"].upper(), "picks": picks,
                     "dev_t": k.get("dev_t"), "dev_other": k.get("dev_other"),
                     "scored": any(picks[q] for q in (1, 2)),
                     "notes": notes.get(g, [])})

    def tally(sub, q):
        t = sum(1 for r in sub if r["picks"][q] == "T")
        o = sum(1 for r in sub if r["picks"][q] == "O")
        return t, o

    out = {"n_groups": len(rows),
           "n_scored": sum(1 for r in rows if r["scored"]),
           "by_pair": {}}
    for pair in ("A", "D"):
        sub = [r for r in rows if r["pair"] == pair]
        t = o = 0
        for q in (1, 2):
            a, b = tally(sub, q)
            t += a
            o += b
        n = t + o
        out["by_pair"][pair] = {"n": n, "t_wins": t, "other_wins": o,
                               "rate": (t / n) if n else 0.0,
                               "p": binom_two_sided(t, n)}

    # P4：逐条 dev 差（对方 dev − T dev，正值代表"离人更远的是对方"）vs 人是否选 T
    xs, ys = [], []
    for r in rows:
        if not r["scored"] or r["dev_t"] is None or r["dev_other"] is None:
            continue
        vote = sum(1 if r["picks"][q] == "T" else -1 for q in (1, 2) if r["picks"][q])
        xs.append(r["dev_other"] - r["dev_t"])
        ys.append(vote)
    rho = spearman(xs, ys)
    out["p4"] = {"n": len(xs), "spearman": rho}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    print("groups %d  scored %d" % (out["n_groups"], out["n_scored"]))
    for pair, t in out["by_pair"].items():
        print("T vs %s: T 胜 %d/%d = %.1f%%  p=%.4f"
              % (pair, t["t_wins"], t["n"], 100 * t["rate"], t["p"]))
    pa = out["by_pair"]["A"]["rate"]
    pd = out["by_pair"]["D"]["rate"]
    print("\nP1 (T vs A >= 65%%): %s" % ("PASS" if pa >= 0.65 else "FAIL"))
    print("P3 (T vs D, T 高出 >=15pp): %s  (%.1fpp)"
          % ("PASS" if (pa - pd) >= 0.15 else "FAIL", 100 * (pa - pd)))
    if rho is None:
        print("P4: 样本不足")
    else:
        print("P4 (per-item dev 差 vs 人选择, Spearman >= 0.30): %s  rho=%.2f  n=%d"
              % ("PASS" if rho >= 0.30 else "FAIL", rho, len(xs)))
    skipped = [r["group"] for r in rows if not r["scored"]]
    if skipped:
        print("skipped:", skipped)
    for r in rows:
        if r["notes"]:
            print("  #%d note: %s" % (r["group"], " / ".join(r["notes"])))


if __name__ == "__main__":
    main()
