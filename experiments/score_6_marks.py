"""Score the 2AFC sheet when the judge marks choices inline.

The judge does not need to produce answers.json: writing ``Q1`` / ``Q2`` / ``Q3``
right after the option he picked in the markdown sheet is enough. This parser
reads those marks, maps them back through key.json, and reports how often the
"adapted" arm beat the "raw" arm.

Usage:
    python experiments/score_6_marks.py data/val/marked.md --key data/val/key.json

A group with no marks is treated as "passed" (not scored) and listed so the
skipped ones stay visible in the record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

GROUP_RE = re.compile(r"^##\s+(\d+)\s*$")
OPT_RE = re.compile(r"^\*\*([12])\*\*\s*$")
MARK_RE = re.compile(r"Q([123])")
TEMPLATE_RE = re.compile(r"^q1\s*\(\s*\)\s*q2\s*\(\s*\)\s*q3\s*\(\s*\)\s*$")


def parse(md_path):
    """Return {group: {1: {marks}, 2: {marks}}} plus free-text notes.

    Option text is the first paragraph of a block. Anything the judge writes
    after that paragraph inside the block, or after the blank answer template
    of the group, is a note.
    """
    groups = {}
    notes = {}
    cur = None
    side = None
    after_template = False
    para_index = 0
    for raw in open(md_path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = GROUP_RE.match(line)
        if m:
            cur = m.group(1)
            groups[cur] = {1: set(), 2: set()}
            side = None
            after_template = False
            para_index = 0
            continue
        if cur is None:
            continue
        if TEMPLATE_RE.match(line.strip()):
            after_template = True
            continue
        m = OPT_RE.match(line)
        if m:
            side = int(m.group(1))
            para_index = 0
            continue
        text = line.strip()
        if not text:
            continue
        if after_template:
            if not set(text) <= set("-—"):
                notes.setdefault(cur, []).append(text)
            continue
        if side is None:
            continue
        marks_found = MARK_RE.findall(text)
        for q in marks_found:
            groups[cur][side].add(int(q))
    return groups, notes


def choice_for(group_marks, q):
    """Which side the judge picked for question q: 1, 2, or None."""
    sides = [s for s, marks in group_marks.items() if q in marks]
    if len(sides) == 1:
        return sides[0]
    return None


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(n + 1):
        pmf = math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        if abs(i - n * p) >= abs(k - n * p) - 1e-12:
            total += pmf
    return min(1.0, total)


def _tally(rows, q):
    c = a = 0
    for r in rows:
        v = r["picks"].get(q)
        if v == "C":
            c += 1
        elif v == "A":
            a += 1
    n = c + a
    return {"c": c, "a": a, "n": n,
            "rate": (c / n) if n else 0.0,
            "p": binom_two_sided(c, n)}


def _pooled(t1, t2):
    c, a = t1["c"] + t2["c"], t1["a"] + t2["a"]
    n = c + a
    return {"c": c, "a": a, "n": n,
            "rate": (c / n) if n else 0.0,
            "p": binom_two_sided(c, n)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--key", default="data/val/key.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    key = {str(k): v for k, v in json.load(open(args.key, encoding="utf-8")).items()}
    groups, notes = parse(args.sheet)

    rows = []
    for g in sorted(groups, key=lambda s: int(s)):
        marks = groups[g]
        k = key.get(g)
        if k is None:
            continue
        c_side = int(k["c_side"])
        a_side = 3 - c_side
        picks = {}
        for q in (1, 2, 3):
            s = choice_for(marks, q)
            if s is None:
                picks[q] = None
            elif s == c_side:
                picks[q] = "C"
            else:
                picks[q] = "A"
        scored = any(picks[q] for q in (1, 2))
        rows.append({"group": int(g), "picks": picks, "scored": scored,
                     "c_side": c_side, "notes": notes.get(g, [])})

    out = {
        "n_groups": len(rows),
        "n_scored": sum(1 for r in rows if r["scored"]),
        "n_skipped": sum(1 for r in rows if not r["scored"]),
        "q1": _tally(rows, 1),
        "q2": _tally(rows, 2),
        "q3": _tally(rows, 3),
        "rows": rows,
    }
    out["q1_q2_pooled"] = _pooled(out["q1"], out["q2"])

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("groups: %d  scored: %d  skipped: %d"
          % (out["n_groups"], out["n_scored"], out["n_skipped"]))
    for q in ("q1", "q2", "q3"):
        t = out[q]
        if t["n"] == 0:
            print("%s: no answers" % q)
            continue
        print("%s: C wins %d/%d = %.1f%%  (A %d, p=%.4f)"
              % (q, t["c"], t["n"], 100 * t["rate"], t["a"], t["p"]))
    p = out["q1_q2_pooled"]
    print("q1+q2 pooled: C wins %d/%d = %.1f%%  (A %d, p=%.4f)"
          % (p["c"], p["n"], 100 * p["rate"], p["a"], p["p"]))
    print("\nP1 (C vs A >= 65%):", "PASS" if p["rate"] >= 0.65 else "FAIL")

    skipped = [r["group"] for r in rows if not r["scored"]]
    if skipped:
        print("skipped groups:", skipped)
    for r in rows:
        if r["notes"]:
            print("  #%d note: %s" % (r["group"], " / ".join(r["notes"])))


if __name__ == "__main__":
    main()
