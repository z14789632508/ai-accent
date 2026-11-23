# -*- coding: utf-8 -*-
"""把体检结果翻译成能直接改的开关。

刻意不输出"温暖度 0.83"这种总分：总分会被当成奖励函数去刷，
然后八股就换个计量单位回来。这里只给两类东西：
1. 采样参数区间（可直接填进 API）；
2. 形状卡（可直接粘进 system prompt 的几句硬约束）。
"""

SAMPLING = {
    "chat": {
        "temperature": (1.0, 1.2),
        "presence_penalty": (0.3, 0.5),
        "frequency_penalty": (0.2, 0.4),
        "max_tokens": (60, 80),
    },
    "formal": {
        "temperature": (0.5, 0.8),
        "presence_penalty": (0.0, 0.2),
        "frequency_penalty": (0.0, 0.2),
        "max_tokens": (700, 1200),
    },
}


def _fmt_range(r):
    return "%.1f" % r[0] if r[0] == r[1] else "%.1f–%.1f" % r


# 不是所有采样参数在每家 API 上都有效。2026-11 核实：
# DeepSeek /chat/completions：frequency_penalty 与 presence_penalty 已标 deprecated，
#   "will not take effect if you pass it to the API"；deepseek-flash 非思考模式下
#   top_p 固定 1.0、传了也被忽略；temperature 0–2 有效；max_tokens 有效；stop 有效。
# 所以在这类后端上，L1 只剩 temperature / max_tokens 两个真旋钮，
# 形状得靠提示与语料去压，别把力气花在空转的参数上。
PROVIDER_SUPPORT = {
    "generic": {"temperature": True, "top_p": True, "presence_penalty": True,
                "frequency_penalty": True, "max_tokens": True},
    "deepseek": {"temperature": True, "top_p": False, "presence_penalty": False,
                 "frequency_penalty": False, "max_tokens": True},
}


def shape_card(register, report, baseline):
    """生成可粘贴的形状卡。只写"要什么形状"，不写"别用什么词"。"""
    m = baseline["metrics"]

    def mid(k, d=0):
        return round(m[k]["p50"], d) if k in m else d

    if register != "chat":
        return None
    lo = round(m["chunk_len_p10"]["p50"]) if "chunk_len_p10" in m else 2
    hi = round(m["chunk_len_p90"]["p50"]) if "chunk_len_p90" in m else 30
    lines = []
    lines.append("回复形状：长度 %d 到 %d 字之间随便给，长短交错，允许一条只有两三个字；" % (lo, hi))
    lines.append("经常只发一条就停，想补一句再发一条，不要攒成一大段；")
    lines.append("句子可以断在半路，不用每段都收得整整齐齐；")
    lines.append("不是每轮都要问回去，也不是每轮都要收尾总结。")
    return "\n".join(lines)


# 只标"太像模型"的方向；反方向偏离不算问题（比人还随意没什么可改的）
FLAG = {
    ("end_punct_rate", "above"): "句末标点率 {v:.0%}（人类 {mid:.0%}）：每段都收句号是最明显的模型痕迹，max_tokens 压小、允许句子断在半路。",
    ("chunk_len_p50", "above"): "单段长度中位数 {v} 字（人类 {mid} 字）：把一条拆成两条，往往比删字更像人。",
    ("chunk_len_cv", "below"): "整体长度分布太窄（CV {v:.2f}，人类 {mid:.2f}）：有的轮次该长，有的该只有两个字。",
    ("turn_share_3", "above"): "恰好 3 段占 {v:.0%}（人类 {mid:.0%}）：不是禁止三段，是让段数跟着内容走。",
    ("turn_chunks_entropy", "below"): "连发条数太可预测（熵 {v:.2f}，人类 {mid:.2f}）：偶尔只发一条就停。",
    ("within_turn_cv_median", "below"): "段内长度太整齐（CV {v:.2f}，人类 {mid:.2f}）：长短交错，允许参差。",
}


def advise(register, report, baseline, rows, provider="generic"):
    """rows = baseline.compare() 的结果。只出数据和方向，不出总分。"""
    out = {"register": register, "sampling": {}, "ineffective": [], "shape_card": None, "notes": []}
    sup = PROVIDER_SUPPORT.get(provider, PROVIDER_SUPPORT["generic"])
    sp = SAMPLING.get(register)
    if sp:
        for k, v in sp.items():
            if sup.get(k, True):
                out["sampling"][k] = _fmt_range(v)
            else:
                out["ineffective"].append(k)
    if out["ineffective"]:
        out["notes"].append(
            "%s 后端上 %s 不生效（API 已弃用或固定值），别指望它们；形状只能靠提示与语料压。"
            % (provider, " / ".join(out["ineffective"])))
    out["shape_card"] = shape_card(register, report, baseline)
    rows = sorted(rows, key=lambda r: -abs(r["dev"]))
    for r in rows:
        if r["verdict"] == "in":
            continue
        tpl = FLAG.get((r["metric"], r["verdict"]))
        if not tpl:
            continue      # 反方向偏离（比人还随意）不是八股，不提示
        tag = "明显" if abs(r["dev"]) >= 2.0 else "轻微"
        out["notes"].append(
            "[%s %.1f×带宽 · 人类百分位 %.1f%%] " % (tag, abs(r["dev"]), r["pct"])
            + tpl.format(v=r["value"], mid=r["p50"], lo=r["p10"], hi=r["p90"]))
    if not out["notes"]:
        out["notes"].append("核心指标没有出现「比人更整齐」的方向性偏离。")
    return out
