# -*- coding: utf-8 -*-
"""把「一串消息」切成「一问一答」的 pair，喂给 `aiaccent echo`。

输入：每行一条消息的 jsonl，字段可改名（默认 ts / uid / text / conv）：
    {"ts": 1795428918, "uid": "u_aaa", "text": "考砸了", "conv": "friendA"}

输出：每行 {"ctx": 上一段, "reply": 这一段}，只保留 reply 由 --src 说的那些对。

为什么要 burst：人连发三条其实是「一次说话」，逐条配对会把半句话当上下文。
同一个人的消息，间隔 < --gap 秒就并进同一个 burst。
"""

import argparse
import json


def load(path, k_ts, k_uid, k_text, k_conv):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = (r.get(k_text) or "").strip()
            if not t:
                continue
            rows.append({"ts": float(r.get(k_ts) or 0), "uid": str(r.get(k_uid) or ""),
                         "text": t, "conv": str(r.get(k_conv) or "_")})
    return rows


def bursts(rows, gap):
    """按会话分组 → 按时间排 → 同人相邻且间隔 < gap 合并。"""
    per = {}
    for r in rows:
        per.setdefault(r["conv"], []).append(r)
    out = []
    for conv, rs in per.items():
        rs.sort(key=lambda x: x["ts"])
        cur = None
        for r in rs:
            if cur and cur["uid"] == r["uid"] and r["ts"] - cur["ts"] <= gap:
                cur["text"] += "\n" + r["text"]
                cur["ts"] = r["ts"]
            else:
                if cur:
                    out.append(cur)
                cur = {"conv": conv, "uid": r["uid"], "ts": r["ts"], "text": r["text"]}
        if cur:
            out.append(cur)
    out.sort(key=lambda x: (x["conv"], x["ts"]))
    return out


def pairs(bs, src=None, ctx_src=None):
    out = []
    prev = {}
    for b in bs:
        p = prev.get(b["conv"])
        prev[b["conv"]] = b
        if not p or p["uid"] == b["uid"]:
            continue
        if src and b["uid"] != src:
            continue
        if ctx_src and p["uid"] != ctx_src:
            continue
        out.append({"ctx": p["text"], "reply": b["text"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="messages jsonl → pairs jsonl（给 aiaccent echo 用）")
    ap.add_argument("messages")
    ap.add_argument("-o", "--out", default="pairs.jsonl")
    ap.add_argument("--gap", type=float, default=120, help="同人连发合并的秒数（默认 120）")
    ap.add_argument("--src", help="只保留这个人说的回复（uid）")
    ap.add_argument("--ctx-src", help="只保留这个人提供的上下文（uid）")
    ap.add_argument("--ts", default="ts")
    ap.add_argument("--uid", default="uid")
    ap.add_argument("--text", default="text")
    ap.add_argument("--conv", default="conv")
    args = ap.parse_args(argv)

    rows = load(args.messages, args.ts, args.uid, args.text, args.conv)
    bs = bursts(rows, args.gap)
    ps = pairs(bs, src=args.src, ctx_src=args.ctx_src)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in ps:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("messages %d → bursts %d → pairs %d → %s" % (len(rows), len(bs), len(ps), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
