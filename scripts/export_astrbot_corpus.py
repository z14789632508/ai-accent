# -*- coding: utf-8 -*-
"""从 AstrBot 的 data_v4.db 导出机器人真实回复语料（jsonl，每行一个 turn）。

用法:
    python scripts/export_astrbot_corpus.py --db path/to/data_v4.db --out data/mine.jsonl

约定（与 aiaccent 对齐）:
- turn = 一条 assistant 消息（模型的"一次回复"）
- chunk = 回复内部按空行/换行切出的段
- 只导出文本，图片/语音等非文本块跳过
"""

import argparse
import json
import os
import re
import sqlite3

SKIP_TYPES = {"image", "record", "video", "file", "face", "at", "reply"}


def blocks_to_text(content):
    """content 可能是 list[dict] 或 str。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            txt = b.get("text")
            if txt is None and isinstance(b.get("data"), dict):
                txt = b["data"].get("text")
            if txt:
                out.append(txt)
        elif t in SKIP_TYPES:
            continue
    return "\n".join(out)


def split_chunks(text):
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # 先按空行切；没有空行就按单换行切（QQ 机器人常用单换行分段）
    parts = re.split(r"\n\s*\n", text)
    if len(parts) == 1:
        parts = text.split("\n")
    return [p.strip() for p in parts if p.strip()]


def looks_like_text(s):
    if len(s) < 1:
        return False
    if s.startswith("[") and s.endswith("]"):
        return False  # CQ 码 / 系统提示
    if "```" in s:
        return False  # 代码块不进人味语料
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(args.db), uri=True)
    rows = con.execute("select user_id, content from conversations").fetchall()

    turns, seen = [], set()
    n_conv = 0
    for user_id, raw in rows:
        if not raw:
            continue
        n_conv += 1
        try:
            msgs = json.loads(raw)
        except Exception:
            continue
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            text = blocks_to_text(m.get("content"))
            if not text.strip():
                continue
            chunks = [c for c in split_chunks(text) if looks_like_text(c)]
            if not chunks:
                continue
            key = "\n".join(chunks)
            if args.dedupe and key in seen:
                continue
            seen.add(key)
            turns.append(chunks)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    n_chunks = sum(len(t) for t in turns)
    print("conversations: %d" % n_conv)
    print("turns: %d  chunks: %d  avg chunks/turn: %.2f"
          % (len(turns), n_chunks, n_chunks / len(turns) if turns else 0))
    print("out: %s" % args.out)


if __name__ == "__main__":
    main()
