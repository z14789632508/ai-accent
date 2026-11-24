# -*- coding: utf-8 -*-
"""适配：从一个人的历史消息里读出 3 条结构指令。

立论（用户给的，比"去八股"准）：
    八股不是错，用不适配的八股才是错。

两条自我约束，写进代码而不是写进文档：

1. **只调结构，不调表皮**。这里不产出"学他说'害''其实'"这类指令。
   模仿用词是谄媚：用户几秒钟就能听出来，然后说"你别学我说话"。
   `profile.fingerprint()` 挖出来的 n-gram **只用于诊断**（他有没有口癖、
   助手有没有继承），永远不进 prompt。

2. **措辞是"这个人习惯 X，所以输出用 Y 形态"，不是"模仿这个人"**。
   前者让模型调参，后者让模型演戏——演戏会过火，过火的标志就是被用户识破。

产出的条数上限是 3：多了模型会挑一条执行、其余当噪声。
"""

from .metrics import measure

# (指标, 条件, 特征说法, 指令)
RULES = [
    ("chunk_len_p50", "short",
     "说话短，一句一个意思",
     "一句一意：不用长复合句，一句话说一件事，说完就停。"),
    ("turn_chunks_median", "many",
     "一次会连发两三条，而不是攒成一大段",
     "允许分条：需要补充时再发一条短消息，不要合并成一大段。"),
    ("hedge_per_kchar", "low",
     "很少留缓冲词（不过/可能/大概），结论直接给",
     "先给结论再给理由：不要用「不过」「可能」「仅供参考」这类词垫在前面。"),
    ("end_punct_rate", "low",
     "几乎不写句号",
     "句末不用句号，句子可以断在半路。"),
    ("chunk_len_cv", "high",
     "长短差得很开，有的三五个字，有的整段",
     "长短交错：不要每条都差不多长，允许一条只有两三个字。"),
]


def _hit(name, thresh, rep):
    v = rep.get(name)
    if v is None:
        return False
    if thresh == "short":
        return v < 10
    if thresh == "many":
        return v >= 3
    if thresh == "low":
        return v < 1.0
    if thresh == "lowp":
        return v < 0.05
    if thresh == "high":
        return v > 1.2
    return False


def advise(turns, max_rules=3):
    """返回 {features, instructions, card}。只吃结构指标，不吃任何词表。"""
    rep = measure(turns)
    picks = []
    for name, thresh, feature, cmd in RULES:
        t = "lowp" if name == "end_punct_rate" else thresh
        if _hit(name, t, rep):
            picks.append({"metric": name, "value": rep[name],
                          "feature": feature, "instruction": cmd})
    picks = picks[:max_rules]

    lines = []
    for p in picks:
        lines.append("- 这个用户习惯%s（实测 %s = %s），所以输出用这种形态：%s"
                     % (p["feature"], p["metric"], p["value"], p["instruction"]))
    return {
        "n_turns": rep["n_turns"],
        "features": picks,
        "instructions": lines,
        "card": "\n".join(lines) if lines else "（这个人的语料没触发任何形态规则，别硬套）",
    }
