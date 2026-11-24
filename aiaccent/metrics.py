# -*- coding: utf-8 -*-
"""ai-accent metrics: measure the *shape* of a text corpus, not its vocabulary.

一个 corpus 由若干 turn 组成，一个 turn 是一个人（或一次模型回复）连续发出的
若干 chunk。模型回复通常一条消息里分好几段 → chunk；真人聊天通常连发好几条
消息 → chunk。

所有指标都在 chunk / turn 两个层级上算，避免"词表"这种一禁就漂移的东西。
"""

import math
import statistics
from collections import Counter

# 句末标点：真人聊天里绝大多数的消息是不打的
END_PUNCT = "。！？!?…~～.…」』）)】”\"'"

# chunk 长度低于这个值不计入 within-turn CV（太短没有形状可言）
MIN_CHUNK_FOR_SHAPE = 2

# 对冲词（hedge）：模型的"安全感"语气——先给结论再补一层"不过/可能/建议"。
# 词表内不做前缀重叠（有"建议"就不再放"建议你"），否则一处命中算两次。
HEDGE = (
    "不过", "建议", "仅供参考", "请务必", "一般来说", "在一般情况下",
    "通常情况下", "如有需要", "需要注意的是", "值得注意的是",
    "或许", "大概", "也许", "似乎", "应该", "通常", "可能",
    "酌情", "视情况",
)


def count_hedges(chunk):
    return sum(chunk.count(w) for w in HEDGE)


def _cv(xs):
    xs = [x for x in xs if x > 0]
    if len(xs) < 2:
        return 0.0
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m else 0.0


def _quantiles(xs, ps=(0.1, 0.5, 0.9)):
    a = sorted(xs)
    return [a[min(len(a) - 1, int(len(a) * p))] for p in ps]


def entropy(counter):
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counter.values() if v)


def ends_with_punct(chunk):
    s = chunk.rstrip()
    return bool(s) and s[-1] in END_PUNCT


def measure(turns, sentence_bin=10):
    """turns: list[list[str]] —— 返回扁平 dict，键名与 baselines/*.json 一致。"""
    turns = [[c for c in t if c and c.strip()] for t in turns]
    turns = [t for t in turns if t]
    chunks = [c for t in turns for c in t]
    if not chunks:
        raise ValueError("empty corpus")

    lens = [len(c) for c in chunks]
    p10, p50, p90 = _quantiles(lens)

    turn_sizes = [len(t) for t in turns]
    size_dist = Counter(turn_sizes)

    within = []
    for t in turns:
        ls = [len(c) for c in t if len(c) >= MIN_CHUNK_FOR_SHAPE]
        if len(ls) >= 2:
            within.append(_cv(ls))

    # 句长：按标点切，bin 显式记录（跨语料比较必须同 bin）
    sent_lens = []
    for c in chunks:
        cur = 0
        for ch in c:
            if ch in END_PUNCT or ch == "\n":
                if cur:
                    sent_lens.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            sent_lens.append(cur)
    sent_ent = entropy(Counter(x // sentence_bin for x in sent_lens)) if sent_lens else 0.0

    # 对冲密度：每千字命中次数 + 含对冲词的 chunk 占比。
    # 为什么要两个：次数密度对"话痨 but 每次只放一个对冲词"敏感，占比对
    # "短句里也塞一句'不过'"敏感——模型的病是后者（见 README 实测）。
    total_chars = sum(len(c) for c in chunks) or 1
    hedge_hits = 0
    hedge_chunk = 0
    for c in chunks:
        n = count_hedges(c)
        hedge_hits += n
        if n:
            hedge_chunk += 1

    return {
        "n_turns": len(turns),
        "n_chunks": len(chunks),
        "chunk_len_p10": p10,
        "chunk_len_p50": p50,
        "chunk_len_p90": p90,
        "chunk_len_mean": round(statistics.fmean(lens), 2),
        "chunk_len_cv": round(_cv(lens), 3),
        "end_punct_rate": round(sum(ends_with_punct(c) for c in chunks) / len(chunks), 3),
        "hedge_per_kchar": round(hedge_hits * 1000 / total_chars, 2),
        "hedge_chunk_share": round(hedge_chunk / len(chunks), 3),
        "turn_chunks_median": statistics.median(turn_sizes),
        "turn_chunks_mean": round(statistics.fmean(turn_sizes), 2),
        "turn_share_1": round(size_dist[1] / len(turns), 3),
        "turn_share_3": round(size_dist[3] / len(turns), 3),
        "turn_chunks_entropy": round(entropy(size_dist), 3),
        "within_turn_cv_median": round(statistics.median(within), 3) if within else 0.0,
        "sentence_len_entropy": round(sent_ent, 3),
        "sentence_bin": sentence_bin,
    }


def bootstrap_ci(turns, names, reps=400, alpha=0.05, seed=7, sentence_bin=10):
    """对已测语料做 turn 级 bootstrap，返回 {metric: (lo, hi)}。

    用途：报告里那个偏离量本身也有噪声（300–500 个 turn 的 CV 估计并不稳），
    给出置信区间，读者才能判断"差 0.05"是不是真的差。
    """
    import random
    rnd = random.Random(seed)
    n = len(turns)
    acc = {k: [] for k in names}
    for _ in range(reps):
        r = measure([turns[rnd.randrange(n)] for _ in range(n)], sentence_bin=sentence_bin)
        for k in names:
            if k in r:
                acc[k].append(r[k])
    out = {}
    for k, xs in acc.items():
        if len(xs) < 10:
            continue
        a = sorted(xs)
        lo = a[int(len(a) * alpha / 2)]
        hi = a[min(len(a) - 1, int(len(a) * (1 - alpha / 2)))]
        out[k] = (lo, hi)
    return out


# 指标方向：+1 越大越像人，-1 越小越像人，0 看区间
DIRECTION = {
    "chunk_len_p50": 0,
    "chunk_len_cv": 0,
    "end_punct_rate": -1,
    "hedge_per_kchar": -1,
    "hedge_chunk_share": -1,
    "turn_chunks_median": 0,
    "turn_share_3": -1,
    "turn_chunks_entropy": 1,
    "within_turn_cv_median": 1,
    "sentence_len_entropy": 1,
}
