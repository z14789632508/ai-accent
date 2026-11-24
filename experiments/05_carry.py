# -*- coding: utf-8 -*-
"""实验 05：承接率——回复有没有接住用户说的**具体东西**。

背景：现有指标量的是形态（长度/条数/标点/对冲）。形态可以刷到全合格，
而回复里一个具体名词都没有。那一类失败现在完全看不见。

做法（全部可复现，无分词依赖）：
1. 用 52 万条群聊统计二元字组的"通用度"，取最常见的一批当停用表。
2. 一条用户消息去掉通用二元组后剩下的，就是它的"具体内容"（专名、数字、动作）。
3. 承接率 = 回复里出现了其中几个 / 一共几个。
   - carry_rate：按上下文里"具体二元组"个数加权（一个人说了 5 个具体点，接住 1 个 = 0.2）
   - carry_hit ：回复里至少接住一个的比例（更粗，但不受上下文长度影响）

对照组：
- bot  ：用户 → 大肥鱼（同一批 QQ 私聊）
- human：人与人的私聊（同一份库，peer != 机器人）

跑法：
  python experiments/05_carry.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CORPUS = r"D:\DSH\shared\corpus\qq"
GROUP = os.path.join(CORPUS, "group_msgs.jsonl")
DM_HUMAN = os.path.join(CORPUS, "dm_human.jsonl")
DM_BOT = os.path.join(CORPUS, "dm_bot.jsonl")
BOT_QQ = 1789681167
GAP = 120          # 同一个人 GAP 秒内连发算一个 burst
TOP_COMMON = 2000  # 群聊里最常见的 N 个二元组，算"通用词"


def bigrams(text):
    t = "".join(ch for ch in text if not ch.isspace())
    return [t[i:i + 2] for i in range(len(t) - 1)]


def load_common():
    """只收纯汉字二元组。

    第一版没过滤，最常见的几个是 `\x0e>`、`",`、`$ǿ`、`*1` —— 机器人消息和代码残留。
    表一脏，carry_rate 的分母就偏：真正具体的专名会被当"通用词"剔掉。
    """
    c = Counter()
    with open(GROUP, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for g in bigrams(r.get("t") or ""):
                if all("\u4e00" <= ch <= "\u9fff" for ch in g):
                    c[g] += 1
    return set(g for g, _ in c.most_common(TOP_COMMON))


def load_events():
    ev = []   # (ts, uid, peer, text)
    for p in (DM_HUMAN, DM_BOT):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                t = (r.get("t") or "").strip()
                if t:
                    ev.append((int(r["ts"]), r["uid"], int(r["peer"]), t))
    ev.sort()
    return ev


def bursts(ev):
    """按 (会话, 说话人) 合并成 burst。会话 key 用 peer 号（私聊里双方一致的那一刻由事件顺序保证）。"""
    out, cur, key = [], [], None
    for ts, uid, peer, t in ev:
        k = (peer, uid)
        if cur and (k != key or ts - cur[-1][0] > GAP):
            out.append((key[0], key[1], [x[1] for x in cur]))
            cur = []
        cur.append((ts, t))
        key = k
    if cur:
        out.append((key[0], key[1], [x[1] for x in cur]))
    return out


def pairs_for(bs, conv, src=None):
    """同一会话里相邻两个不同说话人的 burst → (ctx, reply)。

    src 给定时只保留"由 src 发起"的那些对（问的是 src 的说话习惯）。
    不分方向会把"人的回答"算进机器人的账里。
    """
    seq = [b for b in bs if b[0] == conv]
    out = []
    for a, b in zip(seq, seq[1:]):
        if a[1] == b[1]:
            continue
        if src is not None and a[1] != src:
            continue
        out.append(("".join(a[2]), "".join(b[2])))
    return out


def score(pairs, common):
    """返回 承接率（含复读）、复读率、净承接率（去掉复读后的命中）。"""
    n = hit = echo_n = 0
    ratios = []
    for ctx, rep in pairs:
        ctx_bg = [g for g in bigrams(ctx) if g not in common]
        if not ctx_bg:
            continue
        uniq = set(ctx_bg)
        rep_bg = set(bigrams(rep))
        got = len(uniq & rep_bg)
        # 复读：回复里出现了上下文的一段 4 连字（≥4 字的逐字照搬）
        echo = False
        r = "".join(ch for ch in rep if not ch.isspace())
        c = "".join(ch for ch in ctx if not ch.isspace())
        for i in range(len(c) - 3):
            if c[i:i + 4] in r:
                echo = True
                break
        if echo:
            echo_n += 1
            got = 0          # 复读不算接住
        n += 1
        hit += 1 if got else 0
        ratios.append(got / len(uniq))
    if not n:
        return None
    return {"n": n, "carry_rate": round(sum(ratios) / n, 3),
            "carry_hit": round(hit / n, 3), "echo_rate": round(echo_n / n, 3)}


BUCKETS = [(1, 8), (9, 20), (21, 10 ** 6)]


def score_buckets(pairs, common):
    """按上下文长度分层：任务型长指令和闲聊短句不能混在一起比。"""
    out = []
    for lo, hi in BUCKETS:
        sub = [(c, r) for c, r in pairs if lo <= len(c) <= hi]
        r = score(sub, common)
        out.append(((lo, hi), r))
    return out


def main():
    common = load_common()
    ev = load_events()
    bs = bursts(ev)

    bot_pairs = pairs_for(bs, BOT_QQ, src="u_1i54LHtpjUYBGkgU_1rQLg")
    convs = sorted(set(b[0] for b in bs) - {BOT_QQ})
    human_pairs = []
    for c in convs:
        human_pairs.extend(pairs_for(bs, c))

    if "--dirs" in sys.argv:
        for ts, uid, peer, t in ev[:40]:
            pass
        print("会话数 bot=%d human=%d" % (len(bs), len(convs)))

    print("通用二元组停用表 %d 条（群聊 52 万条取 top%d）" % (len(common), TOP_COMMON))
    for name, ps in (("bot ", bot_pairs), ("human", human_pairs)):
        print("%s 整体 %s" % (name, score(ps, common)))
    print("\n按上下文长度分层（任务型长指令 vs 闲聊短句必须分开看）")
    print("%-12s %-28s %-28s" % ("上下文长度", "bot", "human"))
    bp = dict(score_buckets(bot_pairs, common))
    hp = dict(score_buckets(human_pairs, common))
    for key in bp:
        b, h = bp[key], hp.get(key)
        fmt = lambda r: ("n=%-4d rate=%.3f hit=%.3f echo=%.3f"
                         % (r["n"], r["carry_rate"], r["carry_hit"], r["echo_rate"])) if r else "—"
        print("%-12s %-42s %-42s" % ("%d-%s" % (key[0], "∞" if key[1] > 99999 else key[1]), fmt(b), fmt(h)))

    # 例：接住/没接住各看三条
    print("\n== 例子：接了 ==")
    k = 0
    for ctx, rep in bot_pairs:
        ctx_bg = [g for g in bigrams(ctx) if g not in common]
        if ctx_bg and set(ctx_bg) & set(bigrams(rep)):
            print("  用户: %s\n  机回: %s\n" % (ctx[:40], rep[:60]))
            k += 1
            if k >= 2:
                break
    print("== 例子：没接 ==")
    k = 0
    for ctx, rep in bot_pairs:
        ctx_bg = [g for g in bigrams(ctx) if g not in common]
        if ctx_bg and not (set(ctx_bg) & set(bigrams(rep))):
            print("  用户: %s\n  机回: %s\n" % (ctx[:40], rep[:60]))
            k += 1
            if k >= 2:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
