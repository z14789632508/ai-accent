# -*- coding: utf-8 -*-
"""实验 01：禁词表到底改变了什么（DeepSeek 后端实跑）。

A 组：中性系统提示，不含任何禁令
B 组：同一提示 + 禁用词表（现实世界里大家就是这么干的）
采样参数完全一致，prompt 池相同，各 300 条。

用法：python ab_wordlist.py [--n 300] [--workers 12]
产物：data/ab/a.jsonl、data/ab/b.jsonl（每行一个 turn 的段列表）、data/ab/raw.json
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 仓库根
OUT = os.path.join(ROOT, "data", "ab")

SYS_A = "你是一个中文聊天伙伴。用自然的中文和对方说话。"

BANNED = ["总的来说", "总而言之", "综上所述", "首先", "其次", "再次", "最后",
          "希望对你有帮助", "希望对你有所帮助", "值得一提的是", "需要注意的是",
          "值得注意的是", "另一方面", "此外", "换句话说", "让我们一起", "总结一下"]

SYS_B = ("你是一个中文聊天伙伴。用自然的中文和对方说话。\n"
         "严禁使用以下词语：" + "、".join(BANNED) + "。")

PARAMS = {"model": MODEL, "temperature": 1.0, "max_tokens": 400,
          "thinking": {"type": "disabled"}}


def api_key():
    """key 来源优先级：环境变量 DEEPSEEK_API_KEY > AstrBot 配置（AB_ASTRBOT_CONFIG）。"""
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env:
        return env
    p = os.environ.get("AB_ASTRBOT_CONFIG")
    if not p:
        raise SystemExit("set DEEPSEEK_API_KEY, or AB_ASTRBOT_CONFIG to AstrBot's cmd_config.json")
    cfg = json.load(open(p, encoding="utf-8"))
    for s in cfg.get("provider_sources", []):
        if isinstance(s, dict) and s.get("id") == "deepseek":
            k = s.get("key") or s.get("api_key")
            return k[0] if isinstance(k, list) else k
    raise SystemExit("no deepseek key")


def chat(key, system, user, timeout=90):
    body = dict(PARAMS)
    body["messages"] = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            msg = d["choices"][0]["message"]["content"]
            u = d.get("usage", {})
            return msg, u
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return None, {"error": str(e)[:200]}
            time.sleep(1.5 * (attempt + 1))
    return None, {}


def chunks_of(text):
    parts = [p.strip() for p in re.split(r"\n{2,}", text or "") if p.strip()]
    return parts


def word_hits(text):
    return sum(text.count(w) for w in BANNED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    prompts = [json.loads(l.strip())["p"] for l in
               open(os.path.join(OUT, "prompts.jsonl"), encoding="utf-8")]
    prompts = prompts[:args.n]
    key = api_key()
    os.makedirs(OUT, exist_ok=True)

    jobs = [("a", i, prompts[i], SYS_A) for i in range(len(prompts))] + \
           [("b", i, prompts[i], SYS_B) for i in range(len(prompts))]

    results = {}
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(job):
        grp, idx, p, sysmsg = job
        msg, usage = chat(key, sysmsg, p)
        with lock:
            results[(grp, idx)] = (msg, usage)
            done[0] += 1
            if done[0] % 25 == 0:
                print("  %d/%d  %.0fs" % (done[0], len(jobs), time.time() - t0), flush=True)

    threads = []
    for job in jobs:
        while len([t for t in threads if t.is_alive()]) >= args.workers:
            time.sleep(0.05)
        t = threading.Thread(target=work, args=(job,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    raw, turns = {}, {"a": [], "b": []}
    ptok = ctok = 0
    for (grp, idx), (msg, usage) in sorted(results.items()):
        raw["%s:%d" % (grp, idx)] = {"prompt": prompts[idx], "reply": msg, "usage": usage}
        if msg:
            turns[grp].append(chunks_of(msg))
            ptok += usage.get("prompt_tokens", 0)
            ctok += usage.get("completion_tokens", 0)

    for grp in ("a", "b"):
        with open(os.path.join(OUT, "%s.jsonl" % grp), "w", encoding="utf-8") as f:
            for t in turns[grp]:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump(raw, open(os.path.join(OUT, "raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("ok  a=%d b=%d  prompt_tokens=%d completion_tokens=%d  %.0fs"
          % (len(turns["a"]), len(turns["b"]), ptok, ctok, time.time() - t0))

    for grp in ("a", "b"):
        txt = [raw[k]["reply"] or "" for k in raw if k.startswith(grp + ":")]
        hits = sum(word_hits(t) for t in txt)
        chars = sum(len(t) for t in txt)
        print("  %s  word_hits=%d  chars=%d  per1000=%.2f"
              % (grp, hits, chars, hits * 1000 / max(chars, 1)))


if __name__ == "__main__":
    main()
