# -*- coding: utf-8 -*-
"""从真实语料里挖"最像人"的样本，给 few-shot 用。

为什么不是随机抽：随机抽会把一堆通知、一长段复制粘贴抽进来。这里按
"形状像人"打分——条数、段长、段内参差——再按分数取。分数不代表内容好，
只代表这轮的形状接近人类基线。
"""

import statistics

from . import metrics as M


def score_turn(turn, baseline):
    m = baseline["metrics"]
    ls = [len(c) for c in turn]
    if not ls:
        return -9e9, {}
    lo = m["chunk_len_p50"]["p10"]
    hi = m["chunk_len_p50"]["p90"]
    mid = m["chunk_len_p50"]["p50"]

    # 1 段长是否在人类区间内（越靠中越好）
    center = statistics.median(ls)
    len_pen = 0.0 if lo <= center <= hi else min(abs(center - lo), abs(center - hi)) / max(mid, 1)

    # 2 段内参差：CV 越接近人类中位数越好，但单条消息没有 CV
    cvs = m["within_turn_cv_median"]
    if len(ls) >= 2:
        cv = M._cv(ls)
        cv_pen = abs(cv - cvs["p50"]) / max(cvs["p50"], 0.1)
    else:
        cv_pen = 0.0

    # 3 条数：人类绝大多数 1–4 条
    n = len(ls)
    size_pen = 0.0 if 1 <= n <= 4 else (n - 4) * 0.5

    # 4 句末标点：打了句号的样本反而不像聊天
    punct_pen = sum(M.ends_with_punct(c) for c in turn) / n

    score = -(len_pen + cv_pen + size_pen + 1.2 * punct_pen)
    return score, {"len_pen": round(len_pen, 2), "cv_pen": round(cv_pen, 2),
                   "size_pen": round(size_pen, 2), "punct_pen": round(punct_pen, 2)}


def mine(turns, baseline, k=20, max_chars=60, min_chars=4):
    scored = []
    for t in turns:
        if sum(len(c) for c in t) > max_chars * 2 or sum(len(c) for c in t) < min_chars:
            continue
        s, detail = score_turn(t, baseline)
        scored.append((s, t, detail))
    scored.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for s, t, detail in scored:
        key = tuple(t)
        if key in seen:
            continue
        seen.add(key)
        picked.append({"score": round(s, 3), "turn": t, "detail": detail})
        if len(picked) >= k:
            break
    return picked


def as_fewshot(picked, header="以下是同一个人的真实聊天记录，注意每条的长度和停顿："):
    lines = [header]
    for i, p in enumerate(picked, 1):
        lines.append("")
        for c in p["turn"]:
            lines.append(c)
    return "\n".join(lines)
