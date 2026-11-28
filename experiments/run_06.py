# -*- coding: utf-8 -*-
"""实验 06：验收实验的四臂生成与体检。

四臂：
  A 原样      : 只给中性 system
  B 禁词表    : 中性 system + 62 字禁用词句（与 03/04 同一份）
  C 适配      : 中性 system + adapt 从本人真实语料生成的 <=3 条结构指令
  D 随机扰动  : 对 A 的输出做随机结构扰动（不调 API，本地生成）

用法：
  set AB_ASTRBOT_CONFIG=<AstrBot cmd_config.json>
  python experiments/run_06.py --n 120 --workers 12
产物：data/val/{a,b,c,d}.jsonl、data/val/summary.json
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aiaccent.metrics import measure  # noqa: E402
from aiaccent.baseline import load, compare  # noqa: E402
from aiaccent.adapt import advise  # noqa: E402

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
OUT = os.path.join(ROOT, "data", "val")
PROMPTS = os.path.join(ROOT, "data", "ab", "prompts.jsonl")
USER_TURNS = os.path.join(r"D:\DSH\shared\corpus\qq\by_uid", "u_1i54LHtp.jsonl")

SYS_A = "你是一个中文聊天伙伴。用自然的中文和对方说话。"

BANNED = ["总的来说", "总而言之", "综上所述", "首先", "其次", "再次", "最后",
          "希望对你有帮助", "希望对你有所帮助", "值得一提的是", "需要注意的是",
          "值得注意的是", "另一方面", "此外", "换句话说", "让我们一起", "总结一下"]
LINE_B = "严禁使用以下词语：" + "、".join(BANNED) + "。"

PARAMS = {"model": MODEL, "temperature": 1.0, "max_tokens": 400,
          "thinking": {"type": "disabled"}}

# 留出簇：adapt 的规则不碰这些维度，用来抓"过拟合指标"
HOLDOUT = ["turn_share_3", "turn_chunks_entropy", "within_turn_cv_median",
           "sentence_len_entropy", "hedge_chunk_share"]
HOLDOUT_EXTRA = ["echo_rate"]


def api_key():
    p = os.environ.get("AB_ASTRBOT_CONFIG")
    if not p:
        raise SystemExit("set AB_ASTRBOT_CONFIG to AstrBot's cmd_config.json")
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
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return None, {"error": str(e)[:200]}
            time.sleep(1.5 * (attempt + 1))
    return None, {}


def chunks_of(text):
    parts = [p.strip() for p in re.split(r"\n{2,}", text or "") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in (text or "").split("\n") if p.strip()]
    return parts


def perturb(text, rng):
    """随机结构扰动：删标点 / 合并相邻段 / 从逗号处截断。目标是与 C 同等幅度的形状改变。"""
    chunks = chunks_of(text)
    if not chunks:
        return ""
    out = []
    for c in chunks:
        r = rng.random()
        if r < 0.55:                      # 去掉句末标点
            c = c.rstrip("。！!？?…～~.，,")
        elif r < 0.70 and len(c) > 12:    # 从某个逗号处截断
            cut = [m.start() for m in re.finditer("，", c)]
            if len(cut) > 1:
                c = c[:cut[rng.randrange(1, len(cut))]]
        out.append(c)
    if len(out) > 1 and rng.random() < 0.45:  # 合并两段
        i = rng.randrange(len(out) - 1)
        out[i] = out[i] + "，" + out[i + 1].lstrip("，")
        del out[i + 1]
    if len(out) == 1 and len(out[0]) > 40 and rng.random() < 0.35:  # 拆开
        s = out[0]
        cut = [m.start() for m in re.finditer("，", s)]
        if len(cut) > 1:
            p = cut[rng.randrange(1, len(cut))]
            out = [s[:p], s[p + 1:]]
    return "\n".join(x for x in out if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    prompts = [json.loads(l)["p"] for l in open(PROMPTS, encoding="utf-8")][:args.n]
    user_turns = [json.loads(l) for l in open(USER_TURNS, encoding="utf-8")]
    adv = advise(user_turns, max_rules=3)
    sys_c = SYS_A + "\n" + "\n".join(adv["instructions"])
    print("C 臂指令 %d 条：" % len(adv["instructions"]))
    for x in adv["instructions"]:
        print("  -", x)
    print("C 段长度 %d 字 / B 段 %d 字" % (len(sys_c) - len(SYS_A), len(LINE_B)))

    key = api_key()
    os.makedirs(OUT, exist_ok=True)
    SYS = {"a": SYS_A, "b": SYS_A + "\n" + LINE_B, "c": sys_c}

    jobs = [(g, i) for g in "abc" for i in range(len(prompts))]
    results, lock, done, t0 = {}, threading.Lock(), [0], time.time()

    def work(job):
        g, i = job
        msg, usage = chat(key, SYS[g], prompts[i])
        with lock:
            results[job] = (msg, usage)
            done[0] += 1
            if done[0] % 60 == 0:
                print("  %d/%d %.0fs" % (done[0], len(jobs), time.time() - t0), flush=True)

    threads = []
    for job in jobs:
        while len([t for t in threads if t.is_alive()]) >= args.workers:
            time.sleep(0.05)
        t = threading.Thread(target=work, args=(job,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    turns = {g: [] for g in "abcd"}
    raw = {}
    ptok = ctok = 0
    for (g, i) in sorted(results):
        msg, usage = results[(g, i)]
        raw["%s:%d" % (g, i)] = {"prompt": prompts[i], "reply": msg, "usage": usage}
        if not msg:
            continue
        turns[g].append(chunks_of(msg))
        ptok += usage.get("prompt_tokens", 0)
        ctok += usage.get("completion_tokens", 0)

    # D 臂：本地扰动 A 的输出，逐条对位（保证与 C 是同一批 prompt）
    for i in range(len(turns["a"])):
        rng = random.Random(6000 + i)
        turns["d"].append(perturb("\n".join(turns["a"][i]), rng).split("\n"))

    for g in "abcd":
        with open(os.path.join(OUT, "%s.jsonl" % g), "w", encoding="utf-8") as f:
            for t in turns[g]:
                f.write(json.dumps([x for x in t if x], ensure_ascii=False) + "\n")
    json.dump(raw, open(os.path.join(OUT, "raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    base = load("chat")
    summary = {"n_prompts": len(prompts), "prompt_tokens": ptok,
               "completion_tokens": ctok,
               "c_instructions": adv["instructions"], "arms": {}}
    for g in "abcd":
        rep = measure(turns[g])
        rows = compare(rep, base, experimental=True)
        d = {r["metric"]: {"value": r["value"], "pct": r["pct"], "dev": r["dev"],
                           "verdict": r["verdict"]} for r in rows}
        hold = sum(abs(d[m]["dev"]) for m in HOLDOUT if m in d)
        summary["arms"][g] = {"n": len(turns[g]), "report": rep, "rows": d,
                              "holdout_abs_dev": round(hold, 3)}
        print("\n== %s (%d turns) 留出项 |dev| 之和 %.2f ==" % (g.upper(), len(turns[g]), hold))
        for r in rows:
            print("  %-22s %8s pct%5.1f dev%+7.2f %s" %
                  (r["metric"], round(r["value"], 3), r["pct"], r["dev"], r["verdict"]))

    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    a_h = summary["arms"]["a"]["holdout_abs_dev"]
    c_h = summary["arms"]["c"]["holdout_abs_dev"]
    print("\nP2 泛化：C 留出 %.2f / A 留出 %.2f = %.2f（判据 ≤0.70）" %
          (c_h, a_h, c_h / max(a_h, 1e-9)))
    print("生成卷子：python experiments/make_2afc.py build --a data/val/a.jsonl "
          "--c data/val/c.jsonl --out data/val --n 40 --seed 7")


if __name__ == "__main__":
    main()
