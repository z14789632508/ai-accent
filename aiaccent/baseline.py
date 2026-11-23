# -*- coding: utf-8 -*-
"""人类基线：只存统计量，不存原文。

设计规则（写进 README 也写进这里，防止以后自己破例）：
- register 是一等公民，baselines/<register>.json 各自独立，没有默认值；
- 基线来自真实语料，不能用自己造的样本冒充；
- 存的是每个指标的分位（p10–p90 + 101 点分位表），不是一句话结论；
- band 的口径必须写在文件里（band_method），因为它决定"偏离一个带宽"是什么意思。
"""

import bisect
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

# 先别上主表：分箱口径一换就不可比（见 experiments/02）
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


SEVERE = 2.0   # 偏离达到 2 个带宽算"明显"


def _pct(entry, v):
    """v 在人类分布里的百分位（0–100）。有 101 点分位表就用表，否则按三点插值。"""
    q = entry.get("q")
    if q:
        i = bisect.bisect_left(q, v)
        if i <= 0:
            return 0.0
        if i >= len(q):
            return 100.0
        lo, hi = q[i - 1], q[i]
        frac = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return round((i - 1 + frac) * 100.0 / (len(q) - 1), 1)
    p10, p50, p90 = entry["p10"], entry["p50"], entry["p90"]
    if v <= p10:
        return 10.0
    if v >= p90:
        return 90.0
    if v <= p50:
        w = (p50 - p10) or 1e-9
        return round(10 + 40 * (v - p10) / w, 1)
    w = (p90 - p50) or 1e-9
    return round(50 + 40 * (v - p50) / w, 1)


def compare(report, baseline, experimental=False):
    """返回 [{metric, value, p10, p50, p90, pct, dev, verdict}]

    - pct：这个值在人类分布里的百分位。**优先看它**：0.3% 比"偏离 25 个带宽"直观，
      也不受带宽两侧不对称的影响。
    - dev：偏离量，单位是"该侧带宽"（above 用 p90-p50，below 用 p50-p10）。
      两侧带宽本来就不一样宽，所以 dev 只能同向比较，不能跨指标比较。
    """
    names = CORE + (EXPERIMENTAL if experimental else [])
    rows = []
    metrics = baseline["metrics"]
    for name in names:
        if name not in report or name not in metrics:
            continue
        m = metrics[name]
        v = report[name]
        lo, mid, hi = m["p10"], m["p50"], m["p90"]
        verdict, dev = "in", 0.0
        if v > hi:
            verdict = "above"
            dev = (v - mid) / ((hi - mid) or 0.5)
        elif v < lo:
            verdict = "below"
            dev = (v - mid) / ((mid - lo) or 0.5)
        rows.append({
            "metric": name,
            "value": v,
            "p10": lo, "p50": mid, "p90": hi,
            "pct": _pct(m, v),
            "dev": round(dev, 2),
            "verdict": verdict,
        })
    return rows
