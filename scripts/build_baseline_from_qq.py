# -*- coding: utf-8 -*-
"""从导出的 QQ 群聊 jsonl 生成 baselines/chat.json。

输入 jsonl 每行：{"ts": 秒, "g": 群号, "uid": 发送者, "nick": 群名片, "t": 文本}
（导出方式见 experiments/02_baseline_qq_groups.md；原文不进仓库。）

口径：按"说话人"算，不按群算。模型是一张嘴，所以基线也必须是"一张张嘴之间
的分布"——按群取中位数会把几百个人平均成一锅，band 变窄，容易误判。
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aiaccent import metrics as M  # noqa: E402

KEYS = ["chunk_len_p10", "chunk_len_p50", "chunk_len_p90", "chunk_len_cv", "end_punct_rate",
        "hedge_per_kchar", "hedge_chunk_share",
        "turn_chunks_median", "turn_share_1", "turn_share_3", "turn_chunks_entropy",
        "within_turn_cv_median", "sentence_len_entropy"]


def turns_of(rows, gap=120):
    turns, cur, ts, run = [], None, 0, []
    for d in sorted(rows, key=lambda x: x["ts"]):
        if d["uid"] == cur and d["ts"] - ts <= gap:
            run.append(d["t"])
        else:
            if run:
                turns.append(run)
            run = [d["t"]]
        cur, ts = d["uid"], d["ts"]
    if run:
        turns.append(run)
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", default="", help="默认写 baselines/<register>.json")
    ap.add_argument("--min-msgs", type=int, default=300, help="每个说话人至少多少条")
    ap.add_argument("--sentence-bin", type=int, default=10)
    ap.add_argument("--register", default="chat_group")
    ap.add_argument("--source", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "baselines", "%s.json" % args.register)

    by_speaker = collections.defaultdict(list)
    groups = set()
    total = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            groups.add(int(d["g"]))
            total += 1
            by_speaker[(int(d["g"]), d["uid"])].append(d)

    per_speaker = {}
    for key, rows in by_speaker.items():
        if len(rows) < args.min_msgs:
            continue
        rep = M.measure(turns_of(rows), sentence_bin=args.sentence_bin)
        per_speaker["%s:%s" % key] = rep

    if len(per_speaker) < 20:
        raise SystemExit("not enough speakers: %d" % len(per_speaker))

    def q(a, p):
        return a[min(len(a) - 1, int(len(a) * p))]

    def grid(a, n=101):
        """101 点分位表：用来算"这个值在人类分布里的百分位"。"""
        out = []
        for i in range(n):
            pos = (len(a) - 1) * i / (n - 1)
            lo = int(pos)
            hi = min(len(a) - 1, lo + 1)
            frac = pos - lo
            out.append(round(a[lo] + (a[hi] - a[lo]) * frac, 4))
        return out

    metrics = {}
    for k in KEYS:
        a = sorted(r[k] for r in per_speaker.values())
        metrics[k] = {"p10": q(a, .1), "p50": q(a, .5), "p90": q(a, .9), "q": grid(a)}

    out = {
        "register": args.register,
        "band_method": "across_speakers",
        "source": args.source or ("QQ 群聊，%d 个群、%d 个说话人（每人 ≥%d 条），共 %d 条消息"
                                 % (len(groups), len(per_speaker), args.min_msgs, total)),
        "n_speakers": len(per_speaker),
        "n_turns": sum(r["n_turns"] for r in per_speaker.values()),
        "n_chunks": sum(r["n_chunks"] for r in per_speaker.values()),
        "sentence_bin": args.sentence_bin,
        "notes": [
            "band = 说话人之间的 p10/p50/p90；模型是一张嘴，所以基线也按张嘴取分布",
            "chunk = 一条消息；turn = 同一人 120s 内的连发",
            "sentence_len_entropy 为 experimental：对分箱与语料口径极其敏感，不要跨语料直接比",
            "语料原文不外传；本文件只有统计量",
        ],
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("speakers", len(per_speaker), "turns", out["n_turns"], "chunks", out["n_chunks"])
    for k in KEYS:
        print("  %-24s %s" % (k, metrics[k]))


if __name__ == "__main__":
    main()
