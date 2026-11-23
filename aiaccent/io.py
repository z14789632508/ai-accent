# -*- coding: utf-8 -*-
"""读语料。

支持的输入：
1. .jsonl —— 每行一个 turn。行可以是：
   - ["第一段", "第二段"]                 （chunk 列表）
   - {"chunks": ["...", "..."]}
   - {"text": "第一段\\n\\n第二段"}       （按空行切 chunk）
2. .txt —— 空行分隔 turn，turn 内每行一个 chunk。
3. .db  —— 不直接读，请先导出成上面两种（见 scripts/）。

刻意不内置任何平台的抓法：语料是你的，怎么拿到不该由这个库决定。
"""

import json


def _split_paragraphs(text):
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]


def read_corpus(path):
    turns = []
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list):
                    t = [str(x) for x in obj]
                elif isinstance(obj, dict):
                    if "chunks" in obj:
                        t = [str(x) for x in obj["chunks"]]
                    elif "text" in obj:
                        t = _split_paragraphs(obj["text"])
                    else:
                        raise ValueError("unknown turn object: %s" % list(obj))
                else:
                    t = [str(obj)]
                t = [c for c in t if c.strip()]
                if t:
                    turns.append(t)
    else:
        with open(path, encoding="utf-8") as f:
            buf = []
            for line in f:
                s = line.rstrip("\n")
                if s.strip() == "":
                    if buf:
                        turns.append(buf)
                        buf = []
                else:
                    buf.append(s.strip())
            if buf:
                turns.append(buf)
    return turns


def write_turns(turns, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
