# -*- coding: utf-8 -*-
"""个人档案：把"像人"换成"像这个人"。

为什么需要它（这是本项目的判据转折）：
人群带回答的是"你离普通人有多远"，它把每个人都往中点拉。
可完全不带癖好的文字才是最容易被认出来的 AI 文——擦干净剩下的是"无指纹"，
而无指纹本身就是指纹。所以目标不是分越低越好，是分布像某个具体的人：
他的拖沓、口头禅、写到一半改主意、不用句号、一次只发两三个字。

profile 由两部分组成（同一个人，两种问法）：
1. metrics：这个人自己的指标带（对他的 turn 有放回重采样，band_method=bootstrap_turns）；
   回答"你的节奏落在他自己的波动范围里吗"。
2. fingerprint：这个人相对人群**超额使用**的字串（2/3 字 n-gram，按 count*log(ratio) 排）；
   回答"你有没有他的口癖"。指纹不是要求复读，是要求不缺席。

只存统计量，不存原文。
"""

import json
import math
import os
import random
import re

from . import baseline as B
from .metrics import measure

DIR = os.path.join(B.BASE_DIR, "profiles")

_PUNCT = set("。！？!?…~～.，,、；;：:（）()【】[]「」『』“”\"'—-· \n\t")


def path_of(name):
    return os.path.join(DIR, "%s.json" % name)


def load(name):
    p = path_of(name)
    if not os.path.exists(p):
        raise FileNotFoundError("no profile %r (%s)" % (name, p))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_profiles():
    if not os.path.isdir(DIR):
        return []
    return sorted(x[:-5] for x in os.listdir(DIR) if x.endswith(".json"))


def _chunks(turns):
    return [c for t in turns for c in t if c]


def _ngram_counts(turns, ns=(2, 3)):
    counts = {}
    for c in _chunks(turns):
        for n in ns:
            if len(c) < n:
                continue
            for i in range(len(c) - n + 1):
                g = c[i:i + n]
                if all(ch in _PUNCT for ch in g) or " " in g:
                    continue
                counts[g] = counts.get(g, 0) + 1
    return counts


def _kchars(turns):
    return max(1.0, sum(len(c) for c in _chunks(turns)) / 1000.0)


def fingerprint(turns, crowd_turns=None, min_count=15, top=24, min_ratio=1.3):
    """超额使用的字串 = 口癖。没给 crowd 就只按频次（ratio 记 None）。

    稳定性筛（split-half）：把语料按时间顺序劈成前后两半，**两半都要超额**才留下。
    理由：口癖是稳定的（一个"哈哈哈"的人一年后还是"哈哈哈"），话题不是（这周在聊某话题，
    下周在聊别的）。不做这一层筛，排在前面的是"视频""某话题"这类名词，等于把人家的
    谈资当成了口音。
    """
    own = _ngram_counts(turns)
    own_k = _kchars(turns)
    if crowd_turns:
        cr = _ngram_counts(crowd_turns)
        cr_k = _kchars(crowd_turns)
    else:
        cr, cr_k = None, 1.0

    half = len(turns) // 2
    if cr is not None and half >= 10:
        c1 = _ngram_counts(turns[:half])
        c2 = _ngram_counts(turns[half:])
        k1, k2 = _kchars(turns[:half]), _kchars(turns[half:])
    else:
        c1 = c2 = None

    rows = []
    for g, c in own.items():
        if c < min_count:
            continue
        rate = c / own_k
        if cr is None:
            rows.append({"ngram": g, "count": c, "per_kchar": round(rate, 2),
                         "crowd_per_kchar": None, "ratio": None,
                         "score": round(c, 2)})
            continue
        crate = cr.get(g, 0) / cr_k
        ratio = (rate + 0.01) / (crate + 0.01)
        if ratio < min_ratio:
            continue
        if c1 is not None:
            r1 = (c1.get(g, 0) / k1 + 0.01) / (crate + 0.01)
            r2 = (c2.get(g, 0) / k2 + 0.01) / (crate + 0.01)
            if min(r1, r2) < min_ratio:
                continue
            ratio = min(ratio, min(r1, r2))
        rows.append({"ngram": g, "count": c, "per_kchar": round(rate, 2),
                     "crowd_per_kchar": round(crate, 2), "ratio": round(ratio, 2),
                     "score": round(c * math.log(ratio), 2)})
    rows.sort(key=lambda r: -r["score"])

    # 同一段文字里的长 gram 会把短 gram 盖住（"我跟你讲"命中时"跟你"也在），
    # 保留短的那个更能说明"口癖"，所以按 score 走一遍、跳过被已选项包含的 gram。
    kept = []
    for r in rows:
        g = r["ngram"]
        if any(g in k["ngram"] for k in kept):
            continue
        kept.append(r)
        if len(kept) >= top:
            break
    return kept


def build(turns, name, crowd_turns=None, replicates=1000, resample_turns=300,
          band_floor=0.4, seed=7, sentence_bin=10, top=24):
    n_turns = len(turns)
    n_chunks = sum(len(t) for t in turns)
    if n_turns < 30:
        raise ValueError("too few turns (%d) to describe a person" % n_turns)

    rnd = random.Random(seed)
    k = min(resample_turns, n_turns)
    reps = {m: [] for m in B.CORE + B.EXPERIMENTAL}
    for _ in range(replicates):
        sample = [turns[rnd.randrange(n_turns)] for _ in range(k)]
        r = measure(sample, sentence_bin)
        for m in reps:
            reps[m].append(r[m])

    out = {
        "profile": name,
        "kind": "person",
        "band_method": "bootstrap_turns",
        "band_floor": band_floor,
        "n_turns": n_turns,
        "n_chunks": n_chunks,
        "replicates": replicates,
        "resample_turns": resample_turns,
        "sentence_bin": sentence_bin,
        "metrics": {m: B.band(v, band_floor) for m, v in reps.items()},
        "fingerprint": fingerprint(turns, crowd_turns, top=top),
    }
    os.makedirs(DIR, exist_ok=True)
    p = path_of(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return p, out


def fit(turns, prof, min_ratio=0.5):
    """被测语料在这份指纹上的命中情况。

    hit 的判据是"用到了"，不是"用的一样多"：达到这个人自己密度的 50% 就算在。
    复读他的口癖会很难看；真正要防的是"一次都没有"。
    """
    fps = prof.get("fingerprint") or []
    if not fps:
        return {"hit_rate": None, "hits": 0, "total": 0, "rows": []}
    cnt = _ngram_counts(turns)
    k = _kchars(turns)
    rows = []
    for e in fps:
        rate = cnt.get(e["ngram"], 0) / k
        ratio = rate / e["per_kchar"] if e["per_kchar"] else 0.0
        rows.append({"ngram": e["ngram"], "per_kchar": round(rate, 2),
                     "person_per_kchar": e["per_kchar"],
                     "ratio": round(ratio, 2), "hit": ratio >= min_ratio})
    hits = sum(1 for r in rows if r["hit"])
    return {"hit_rate": round(hits / len(rows), 3), "hits": hits,
            "total": len(rows), "rows": rows}
