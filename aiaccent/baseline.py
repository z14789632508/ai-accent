# -*- coding: utf-8 -*-
"""人类基线：只存统计量，不存原文。

设计规则（写进 README 也写进这里，防止以后自己破例）：
- register 是一等公民，baselines/<register>.json 各自独立，没有默认值；
- 基线来自真实语料，不能用自己造的样本冒充；
- 存的是每群/每段的分布带（p10–p90），不是一句话结论。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE_DIR = os.path.join(ROOT, "baselines")

# 参与"是否在人味区间内"判断的指标，以及缺了它就没意义的方向
CORE = [
    "chunk_len_p50",
    "chunk_len_cv",
    "end_punct_rate",
    "turn_chunks_median",
    "turn_share_3",
    "turn_chunks_entropy",
    "within_turn_cv_median",
]

EXPERIMENTAL = ["sentence_len_entropy"]


def path_of(register):
    return os.path.join(BASE_DIR, "%s.json" % register)


def load(register):
    p = path_of(register)
    if not os.path.exists(p):
        raise FileNotFoundError("no baseline for register=%r (%s)" % (register, p))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_registers():
    if not os.path.isdir(BASE_DIR):
        return []
    return sorted(x[:-5] for x in os.listdir(BASE_DIR) if x.endswith(".json"))


SEVERE = 2.0   # 偏离达到 2 个带宽算"明显"：band 宽 = p90-p50 / p50-p10


def compare(report, baseline):
    """返回 [(metric, observed, lo, mid, hi, verdict, dev)]

    dev = 偏离量，单位是"一个带宽"。above 用 (v-p50)/(p90-p50)，
    below 用 (v-p50)/(p50-p10)。在带内为 0。
    方向和幅度都要看：方向说明像不像模型，幅度说明是不是噪声。
    """
    rows = []
    metrics = baseline["metrics"]
    for name in CORE + EXPERIMENTAL:
        if name not in report or name not in metrics:
            continue
        m = metrics[name]
        v = report[name]
        lo, mid, hi = m["p10"], m["p50"], m["p90"]
        dev = 0.0
        if v > hi and hi > mid:
            verdict, dev = "above", (v - mid) / (hi - mid)
        elif v < lo and mid > lo:
            verdict, dev = "below", (v - mid) / (mid - lo)
        else:
            verdict = "in"
        rows.append((name, v, lo, mid, hi, verdict, round(dev, 2)))
    return rows
