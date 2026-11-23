# -*- coding: utf-8 -*-
"""实验 02：多一个等长对照，把"词表效应"和"多了一段提示"分开。

三组，system prompt 只差最后那一段，且 B/C 两段**字符数完全相同**：
  A: 只有中性提示
  B: 中性提示 + 禁用词表
  C: 中性提示 + 与风格无关的等长叮嘱（保持友善、说清楚…）

两个 prompt 池：
  chat  真实聊天里捞出来的 300 条（本来就不怎么说八股词）
  badu  专挑最容易触发八股的 300 条（分析/三条建议/解释/总结/介绍）

要看的三件事：
  1. 词级命中率（每千字命中次数）—— B 到底压住了多少"物质量"；
  2. 结构指标（句末标点率 / 单段占比 / 条数熵 / 段内 CV）—— B 有没有把形状压整齐；
  3. C 组是否跟着一起变整齐 —— 如果 C 也变，那"水床"就只是"多了一段约束"。

用法：
  python ab_v2.py --pools chat,badu --n 300
产物：data/ab2/{pool}_{grp}.jsonl、data/ab2/summary.json
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aiaccent.metrics import measure  # noqa: E402

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
OUT = os.path.join(ROOT, "data", "ab2")
PROMPTS_CHAT = os.path.join(ROOT, "data", "ab", "prompts.jsonl")

SYS_A = "你是一个中文聊天伙伴。用自然的中文和对方说话。"

BANNED = ["总的来说", "总而言之", "综上所述", "首先", "其次", "再次", "最后",
          "希望对你有帮助", "希望对你有所帮助", "值得一提的是", "需要注意的是",
          "值得注意的是", "另一方面", "此外", "换句话说", "让我们一起", "总结一下"]

LINE_B = "严禁使用以下词语：" + "、".join(BANNED) + "。"

# C 组：与"像不像人/八股"完全无关的叮嘱，长度对齐 LINE_B
NEUTRAL = ("回答的时候保持友善和耐心，遇到不确定的事就直说，不要假设对方已经知道前因后果，"
           "也不要把话说满，把能做到的和做不到的分开讲，对方追问时就按他的用法接着答，"
           "不要额外补充他没问的信息，也不要在句尾加上没有意义的客套，多想想他真正在问什么。")

PARAMS = {"model": MODEL, "temperature": 1.0, "max_tokens": 400,
          "thinking": {"type": "disabled"}}


def pad_to(text, n):
    """用整句拼到贴近 n 个字符（允许几字误差，实测记录真实长度）。"""
    s = ""
    while len(s) < n:
        s += text
    s = s[:n]
    cut = max(s.rfind("，"), s.rfind("。"))
    if cut >= n * 0.8:
        s = s[:cut + 1]
    return s


def api_key():
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
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return None, {"error": str(e)[:200]}
            time.sleep(1.5 * (attempt + 1))
    return None, {}


def chunks_of(text):
    # 与真人侧同口径：空行分段；没有空行就按单换行（模型常见"一段一行"）
    parts = [p.strip() for p in re.split(r"\n{2,}", text or "") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in (text or "").split("\n") if p.strip()]
    return parts


def word_hits(text):
    return sum((text or "").count(w) for w in BANNED)


TOPICS = ["远程办公", "健身打卡", "宠物狗拆家", "考研背书", "短视频上瘾", "室友卫生",
          "存款理财", "换工作", "健身减脂", "失眠", "学英语口语", "送礼", "异地恋",
          "开网店", "租房选址", "通勤太远", "做饭省钱", "游戏氪金", "考驾照", "带娃"]

TEMPLATES = [
    "请分析一下{t}这件事的利弊",
    "给出三条关于{t}的建议",
    "解释一下{t}背后的原因",
    "总结一下{t}的要点",
    "介绍一下{t}的正确做法",
    "帮我写一段关于{t}的说明",
    "关于{t}，我该注意什么",
    "为什么大家在{t}上看法差这么多",
]


def pool_badu(n):
    out = []
    i = 0
    while len(out) < n:
        t = TOPICS[i % len(TOPICS)]
        tpl = TEMPLATES[(i // len(TOPICS)) % len(TEMPLATES)]
        out.append(tpl.format(t=t))
        i += 1
    return out


def load_pool(name, n):
    if name == "badu":
        return pool_badu(n)
    if name == "chat":
        ps = [json.loads(l.strip())["p"] for l in open(PROMPTS_CHAT, encoding="utf-8")]
        return ps[:n]
    raise SystemExit("unknown pool %r" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="chat,badu")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    line_c = pad_to(NEUTRAL, len(LINE_B))
    SYS = {
        "a": SYS_A,
        "b": SYS_A + "\n" + LINE_B,
        "c": SYS_A + "\n" + line_c,
    }
    print("B 段 %d 字 / C 段 %d 字（等长）" % (len(LINE_B), len(line_c)))

    key = api_key()
    os.makedirs(OUT, exist_ok=True)
    summary = {"line_b": LINE_B, "line_c": line_c, "pools": {}}

    for pool in args.pools.split(","):
        prompts = load_pool(pool, args.n)
        jobs = [(g, i, prompts[i], SYS[g]) for g in "abc" for i in range(len(prompts))]
        results = {}
        lock = threading.Lock()
        done = [0]
        t0 = time.time()

        def work(job):
            g, idx, p, sysmsg = job
            msg, usage = chat(key, sysmsg, p)
            with lock:
                results[(g, idx)] = (msg, usage)
                done[0] += 1
                if done[0] % 100 == 0:
                    print("  %s %d/%d %.0fs" % (pool, done[0], len(jobs), time.time() - t0),
                          flush=True)

        threads = []
        for job in jobs:
            while len([t for t in threads if t.is_alive()]) >= args.workers:
                time.sleep(0.05)
            t = threading.Thread(target=work, args=(job,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        raw, turns = {}, {g: [] for g in "abc"}
        ptok = ctok = 0
        for (g, idx), (msg, usage) in sorted(results.items()):
            raw["%s:%d" % (g, idx)] = {"prompt": prompts[idx], "reply": msg, "usage": usage}
            if msg:
                turns[g].append(chunks_of(msg))
                ptok += usage.get("prompt_tokens", 0)
                ctok += usage.get("completion_tokens", 0)

        pool_out = {"n_prompts": len(prompts), "prompt_tokens": ptok,
                    "completion_tokens": ctok, "groups": {}}
        for g in "abc":
            with open(os.path.join(OUT, "%s_%s.jsonl" % (pool, g)), "w",
                      encoding="utf-8") as f:
                for t in turns[g]:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            txt = [raw[k]["reply"] or "" for k in raw if k.startswith(g + ":")]
            hits = sum(word_hits(t) for t in txt)
            chars = sum(len(t) for t in txt)
            rep = measure(turns[g])
            pool_out["groups"][g] = {
                "n": len(turns[g]),
                "word_hits": hits,
                "chars": chars,
                "hits_per_1000": round(hits * 1000 / max(chars, 1), 2),
                "report": rep,
            }
        json.dump(raw, open(os.path.join(OUT, "%s_raw.json" % pool), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
        summary["pools"][pool] = pool_out
        print("%s done %.0fs" % (pool, time.time() - t0), flush=True)

    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 打印对照表
    keys = ["hits_per_1000", "end_punct_rate", "turn_share_1", "turn_share_3",
            "turn_chunks_entropy", "within_turn_cv_median", "chunk_len_p50"]
    for pool, d in summary["pools"].items():
        print("\n== %s (%d prompts) ==" % (pool, d["n_prompts"]))
        print("%-24s %8s %8s %8s" % ("metric", "A", "B", "C"))
        for k in keys:
            if k == "hits_per_1000":
                print("%-24s %8.2f %8.2f %8.2f" % (k, d["groups"]["a"][k],
                                                   d["groups"]["b"][k], d["groups"]["c"][k]))
            else:
                print("%-24s %8s %8s %8s" % (k, d["groups"]["a"]["report"][k],
                                             d["groups"]["b"]["report"][k],
                                             d["groups"]["c"]["report"][k]))


if __name__ == "__main__":
    main()
