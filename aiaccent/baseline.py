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
    "hedge_per_kchar",
    "hedge_chunk_share",
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


def is_baseline(obj):
    """是不是一份人群基线（不是指纹库、也不是二元组表）。"""
    return (isinstance(obj, dict) and "n_turns" in obj
            and "n_chunks" in obj and isinstance(obj.get("metrics"), dict)
            and any(k in obj["metrics"] for k in CORE))


def list_registers():
    if not os.path.isdir(BASE_DIR):
        return []
    out = []
    for x in sorted(os.listdir(BASE_DIR)):
        if not x.endswith(".json"):
            continue
        name = x[:-5]
        try:
            if is_baseline(load(name)):
                out.append(name)
        except (ValueError, OSError):
            continue
    return out



SEVERE = 2.0   # 偏离达到 2 个带宽算"明显"


def quantile_list(values, n=101):
    """0/1/2/…/100 分位表。"""
    a = sorted(values)
    if not a:
        return [0.0] * n
    out = []
    for i in range(n):
        pos = (len(a) - 1) * i / (n - 1)
        lo = int(pos)
        hi = min(len(a) - 1, lo + 1)
        frac = pos - lo
        out.append(round(a[lo] + (a[hi] - a[lo]) * frac, 4))
    return out


def band(values, floor=0.0):
    """把一组重复采样得到的分位表包成 {p10,p50,p90,q}。

    floor>0：说话人太少时 bootstrap 只反映"一个人的抽样误差"，不代表"人和人能差多少"。
    所以按 p50 的相对比例设带宽下限——宁可把带放宽，也不制造"±0.5 就明显偏离"的假精度。
    """
    q = quantile_list(values)
    if floor > 0:
        p50 = q[50]
        want = abs(p50) * floor
        lo_half = max(p50 - q[10], want, 1e-9)
        hi_half = max(q[90] - p50, want, 1e-9)
        slo = lo_half / max(p50 - q[10], 1e-9)
        shi = hi_half / max(q[90] - p50, 1e-9)
        q = [round(p50 - (p50 - x) * slo if x < p50 else p50 + (x - p50) * shi, 4) for x in q]
        if q[90] - p50 < 1e-9:
            q = [q[i] if i <= 50 else round(p50 + want * (i - 50) / 50.0, 4) for i in range(101)]
        if p50 - q[10] < 1e-9 and p50 != 0:
            q = [q[i] if i >= 50 else round(p50 - want * (50 - i) / 50.0, 4) for i in range(101)]
    return {"p10": q[10], "p50": q[50], "p90": q[90], "q": q}


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
