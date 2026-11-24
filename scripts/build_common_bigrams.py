# -*- coding: utf-8 -*-
"""从一份大语料里取最常见的二元字组，当「通用词」。

用途：`aiaccent echo --common baselines/common_bigrams.json` 才会算承接率。
没有这份表就只出复读率——那样也够用，而且不用先准备大语料。

    python scripts/build_common_bigrams.py D:\\DSH\\shared\\corpus\\qq\\group_turns.jsonl
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from aiaccent.content import bigrams  # noqa: E402
from aiaccent.io import read_corpus  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("-n", type=int, default=2000, help="保留多少个最常见的二元组")
    ap.add_argument("-o", "--out", default="baselines/common_bigrams.json")
    args = ap.parse_args(argv)

    turns = read_corpus(args.corpus)
    cnt = {}
    n = 0
    for t in turns:
        for chunk in t:
            n += len(chunk)
            for g in bigrams(chunk):
                cnt[g] = cnt.get(g, 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[: args.n]
    out = {"n": args.n, "from": os.path.basename(args.corpus),
           "turns": len(turns), "chars": n, "bigrams": [g for g, _ in top]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote %s  bigrams=%d  chars=%d  (top: %s)"
          % (args.out, len(out["bigrams"]), n, "、".join(out["bigrams"][:8])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
