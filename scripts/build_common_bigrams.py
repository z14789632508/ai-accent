# -*- coding: utf-8 -*-
"""从一份大语料里取最常见的**汉字**二元组，当「通用词」。

用途：`aiaccent echo --common baselines/common_bigrams.json` 才会算承接率。
没有这份表就只出复读率——那样也够用，而且不用先准备大语料。

    python scripts/build_common_bigrams.py D:\\DSH\\shared\\corpus\\qq\\group_turns.jsonl
    python scripts/build_common_bigrams.py D:\\DSH\\shared\\corpus\\qq\\group_msgs.jsonl --field t

为什么只收纯汉字：第一版没过滤，取出来的头几名是 `\\x0e>`、`",`、`$ǿ`、`*1` 这种
机器人消息和代码片段的残留。表一脏，`carry_rate` 的分母就偏——真正具体的专名会被
当成"通用词"剔掉，剩下没内容的反而算"具体"。2000 条里只有 1357 条是汉字，等于一半噪声。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from aiaccent.content import bigrams  # noqa: E402
from aiaccent.io import read_corpus  # noqa: E402


def is_han(bg):
    return all("\u4e00" <= ch <= "\u9fff" for ch in bg)


def iter_texts(corpus, field=None):
    """两种输入：turn 级语料（read_corpus 能吃）或消息级 jsonl（一行一条，--field 指字段）。"""
    if field:
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                t = (r.get(field) or "").strip() if isinstance(r, dict) else ""
                if t:
                    yield t
        return
    for turn in read_corpus(corpus):
        for chunk in turn:
            yield chunk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("-n", type=int, default=2000, help="保留多少个最常见的二元组")
    ap.add_argument("--field", help="消息级 jsonl 的文本字段名（如 t / text）")
    ap.add_argument("--include-non-han", action="store_true", help="不过滤非汉字（不推荐）")
    ap.add_argument("-o", "--out", default="baselines/common_bigrams.json")
    args = ap.parse_args(argv)

    cnt = {}
    n = 0
    for text in iter_texts(args.corpus, args.field):
        n += len(text)
        for g in bigrams(text):
            if not args.include_non_han and not is_han(g):
                continue
            cnt[g] = cnt.get(g, 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[: args.n]
    out = {"n": args.n, "from": os.path.basename(args.corpus),
           "chars": n, "han_only": not args.include_non_han,
           "bigrams": [g for g, _ in top]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote %s  bigrams=%d  chars=%d  (top: %s)"
          % (args.out, len(out["bigrams"]), n, "、".join(out["bigrams"][:10])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
