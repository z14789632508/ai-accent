# -*- coding: utf-8 -*-
"""把 dm_human.jsonl 按说话人切成 turn 级语料，好给每个人建一份档案。

输入：每行 {"ts":..., "uid":"u_xxx", "peer":<qq>, "t":"文本"}
输出：<out-dir>/<uid前8位>.jsonl  每人一个文件，每行是一个 turn（段列表）
     <out-dir>/crowd.jsonl       除主说话人之外的所有人（算指纹超额倍数用）

用法：
  python scripts/split_dm_uid.py D:\\DSH\\shared\\corpus\\qq\\dm_human.jsonl --out D:\\DSH\\shared\\corpus\\qq\\by_uid --main u_1i54LHtpjUYBGkgU_1rQLg
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts.build_baseline_dm import load_rows, to_turns  # noqa: E402


def write_turns(path, turns):
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--main", default="", help="主说话人 uid（其余人合成 crowd）")
    ap.add_argument("--min-msgs", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_rows(args.input)
    turns_of = to_turns(rows)

    counts = {u: sum(len(x) for x in t) for u, t in turns_of.items()}
    crowd = []
    for u, t in turns_of.items():
        if counts[u] < args.min_msgs:
            continue
        write_turns(os.path.join(args.out, "%s.jsonl" % u[:10]), t)
        if u != args.main:
            crowd.extend(t)
    if args.main:
        write_turns(os.path.join(args.out, "crowd.jsonl"), crowd)

    for u, c in sorted(counts.items(), key=lambda x: -x[1]):
        print("%-26s %5d msgs  %4d turns" % (u, c, len(turns_of[u])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
