# -*- coding: utf-8 -*-
"""内容层指标：这段回复到底是「接住了」还是「复述了」。

为什么单独一个模块：形态指标吃一个 turn 列表（`measure(turns)`），
内容指标必须吃**成对**输入 `(ctx, reply)`——没有上一条，就无从判断接没接。

两个量，含义完全不同：

- echo_rate  ：回复里逐字照搬了上下文的一段 4 连字。
               人 3.8%、机器 16.3%（同一批 QQ 私聊，见 experiments/05）。上下文越长机器越爱抄。
               这是「你别学我说话」在数据上的样子。
- carry_rate ：回复里出现了上下文里若干「具体二元组」的多少个。
               注意它**不能单独用**：人在聊天时会换词（"考砸了" → "哪儿没考好"），
               字面重叠接近 0，所以这个值低并不代表没接住；机器复述原词反而高。
               要么配 echo 一起看（复读的那些对计 0），要么走语义度量。
"""

import json


def read_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                continue          # turn 级语料（一行一个列表）不是 pair，直接跳过
            ctx, rep = r.get("ctx") or "", r.get("reply") or ""
            if isinstance(ctx, (list, tuple)):
                ctx = "".join(ctx)
            if isinstance(rep, (list, tuple)):
                rep = "".join(rep)
            if str(ctx).strip() and str(rep).strip():
                out.append((str(ctx), str(rep)))
    return out


def _flat(s):
    return "".join(ch for ch in s if not ch.isspace())


def bigrams(text):
    t = _flat(text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def is_echo(ctx, reply, n=4):
    """回复里是否逐字出现了上下文的一段 n 连字。"""
    c, r = _flat(ctx), _flat(reply)
    if len(c) < n:
        return False
    for i in range(len(c) - n + 1):
        if c[i:i + n] in r:
            return True
    return False


def echo_stats(pairs, n=4, worst=0):
    rows = []
    for ctx, rep in pairs:
        rows.append({"ctx": ctx, "reply": rep, "echo": is_echo(ctx, rep, n)})
    hits = sum(1 for r in rows if r["echo"])
    out = {"n": len(rows), "echo_rate": round(hits / len(rows), 3) if rows else 0.0,
           "echo_n": hits, "min_chars": n}
    if worst:
        out["examples"] = [r for r in rows if r["echo"]][:worst]
    return out


def carry_stats(pairs, common, min_chars=2):
    """common：通用二元组集合（来自大语料）。复读的对计 0。"""
    n = hit = 0
    ratios = []
    for ctx, rep in pairs:
        ctx_bg = [g for g in bigrams(ctx) if g not in common]
        if not ctx_bg:
            continue
        uniq = set(ctx_bg)
        rep_bg = set(bigrams(rep))
        got = len(uniq & rep_bg)
        if is_echo(ctx, rep, 4):
            got = 0
        n += 1
        hit += 1 if got else 0
        ratios.append(got / len(uniq))
    if not n:
        return {"n": 0}
    return {"n": n, "carry_rate": round(sum(ratios) / n, 3),
            "carry_hit": round(hit / n, 3)}
