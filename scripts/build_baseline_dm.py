# -*- coding: utf-8 -*-
"""从私聊语料（每人一行消息）生成 chat_dm 基线。

为什么需要这个：群聊语料里"抢话"会把消息压短（chunk_len_p50 偏低、CV 偏大），
拿它去量一对一私聊的模型回复，等于拿两种语境比。私聊语料才对得上。

输入 jsonl，每行：{"ts": 1710000000, "uid": "u_xxx", "t": "消息文本"}
输出 baselines/<register>.json（只存统计量，不存原文）。

band 的两种口径（写进文件里，别猜）：
- across_speakers：说话人 >= 5 且每人 >= min-msgs，对每个说话人算一份指标，
  再对"说话人之间"取分位 —— 回答"不同的嘴有多不一样"。
- bootstrap_turns：说话人太少（比如只有 1 个人），对 turn 有放回重采样
  replicates 次，每次算一份指标，再取分位 —— 回答"这个人的样本波动有多大"。
  比 across_speakers 窄，但不伪造人数。

用法：
  python scripts/build_baseline_dm.py dm_human.jsonl --register chat --min-msgs 200
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aiaccent.metrics import measure  # noqa: E402
from aiaccent.baseline import CORE, EXPERIMENTAL, BASE_DIR  # noqa: E402

METRICS = CORE + EXPERIMENTAL
GAP = 120          # 同一人 120 秒内连发算一个 turn


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = (r.get("t") or "").strip()
            if not t:
                continue
            rows.append((int(r["ts"]), r["uid"], t))
    return rows


def to_turns(rows, gap=GAP, split_lines=True):
    """按 uid 分组、按时间排序、gap 内合并成 turn。

    split_lines: 一条消息里的多行/多段算多个 chunk。必须开——AstrBot 常常把
    "三段话"塞进一条消息里（用换行分开），不拆的话模型侧的 3 段结构在 chunk
    这一层就看不见了，量出来的"条数"会跟真人侧不是同一个单位。
    """
    by = {}
    for ts, uid, t in rows:
        by.setdefault(uid, []).append((ts, t))
    turns_of = {}
    for uid, seq in by.items():
        seq.sort()
        turns, cur, last = [], [], None
        for ts, t in seq:
            if last is not None and ts - last > gap:
                if cur:
                    turns.append(cur)
                cur = []
            if split_lines:
                cur.extend([c.strip() for c in t.split("\n") if c.strip()])
            else:
                cur.append(t)
            last = ts
        if cur:
            turns.append(cur)
        turns_of[uid] = turns
    return turns_of


def quantile_list(values, n=101):
    a = sorted(values)
    out = []
    for i in range(n):
        pos = (len(a) - 1) * i / (n - 1)
        lo = int(pos)
        hi = min(len(a) - 1, lo + 1)
        frac = pos - lo
        out.append(round(a[lo] + (a[hi] - a[lo]) * frac, 4))
    return out


def band(values, floor=0.0):
    q = quantile_list(values)
    if floor > 0:
        # 说话人太少时，bootstrap 只反映"这一个说话人的抽样误差"，
        # 不反映"人和人之间能差多少"。强制加一条相对带宽下限，
        # 宁可把带放宽，也不制造"±0.5 就明显偏离"的假精度。
        p50 = q[50]
        want = abs(p50) * floor
        lo_half = max(p50 - q[10], want, 1e-9)
        hi_half = max(q[90] - p50, want, 1e-9)
        slo = lo_half / max(p50 - q[10], 1e-9)
        shi = hi_half / max(q[90] - p50, 1e-9)
        q = [round(p50 - (p50 - x) * slo if x < p50 else p50 + (x - p50) * shi, 4) for x in q]
        # 上侧没有离散（比如只有一个人说话时 p50=p90）时，直接铺一条线性上带
        if q[90] - p50 < 1e-9:
            q = [q[i] if i <= 50 else round(p50 + want * (i - 50) / 50.0, 4) for i in range(101)]
        if p50 - q[10] < 1e-9 and p50 != 0:
            q = [q[i] if i >= 50 else round(p50 - want * (50 - i) / 50.0, 4) for i in range(101)]
    return {"p10": q[10], "p50": q[50], "p90": q[90], "q": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--register", default="chat")
    ap.add_argument("--min-msgs", type=int, default=200)
    ap.add_argument("--replicates", type=int, default=1000)
    # 重采样到"评估规模"再算指标：模型那边通常只有 300–500 个 turn，
    # 拿 2700 条语料算出来的 p50 精度过高，会把这个精度当成"人的严格程度"。
    ap.add_argument("--resample-turns", type=int, default=300)
    ap.add_argument("--source", default="")
    ap.add_argument("--sentence-bin", type=int, default=10)
    ap.add_argument("--band-floor", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_rows(args.input)
    turns_of = to_turns(rows)

    voices = {u: t for u, t in turns_of.items()
              if sum(len(x) for x in t) >= args.min_msgs}
    if not voices:
        sys.exit("no voice with >= %d msgs" % args.min_msgs)

    n_speakers = len(voices)
    all_turns = [t for ts in voices.values() for t in ts]
    n_turns = len(all_turns)
    n_chunks = sum(len(t) for t in all_turns)

    if n_speakers >= 5:
        method = "across_speakers"
        per = [measure(t, args.sentence_bin) for t in voices.values()]
        metrics = {m: band([p[m] for p in per]) for m in METRICS}
    else:
        method = "bootstrap_turns"
        rnd = random.Random(args.seed)
        k = min(args.resample_turns, n_turns)
        reps = {m: [] for m in METRICS}
        for _ in range(args.replicates):
            sample = [all_turns[rnd.randrange(n_turns)] for _ in range(k)]
            r = measure(sample, args.sentence_bin)
            for m in METRICS:
                reps[m].append(r[m])
        metrics = {m: band(reps[m], args.band_floor) for m in METRICS}

    out = {
        "register": args.register,
        "source": args.source or os.path.basename(args.input),
        "band_method": method,
        "band_floor": args.band_floor if method == "bootstrap_turns" else None,
        "n_speakers": n_speakers,
        "n_turns": n_turns,
        "n_chunks": n_chunks,
        "sentence_bin": args.sentence_bin,
        "turn_gap_seconds": GAP,
        "replicates": args.replicates if method == "bootstrap_turns" else None,
        "resample_turns": args.resample_turns if method == "bootstrap_turns" else None,
        "metrics": metrics,
    }
    os.makedirs(BASE_DIR, exist_ok=True)
    p = os.path.join(BASE_DIR, "%s.json" % args.register)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", p, "method=%s speakers=%d turns=%d chunks=%d"
          % (method, n_speakers, n_turns, n_chunks))


if __name__ == "__main__":
    main()
