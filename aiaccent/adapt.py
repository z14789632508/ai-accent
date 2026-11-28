# -*- coding: utf-8 -*-
"""适配：从一个人的历史消息里读出结构指令。

立论（用户给的，比"去八股"准）：
    八股不是错，用不适配的八股才是错。

三条自我约束，写进代码而不是写进文档：

1. **只调结构，不调表皮**。这里不产出"学他说'害''其实'"这类指令。
   模仿用词是谄媚：用户几秒钟就能听出来，然后说"你别学我说话"。
   `profile.fingerprint()` 挖出来的 n-gram **只用于诊断**（他有没有口癖、
   助手有没有继承），永远不进 prompt。

2. **措辞是"这个人习惯 X，所以输出用 Y 形态"，不是"模仿这个人"**。
   前者让模型调参，后者让模型演戏——演戏会过火，过火的标志就是被用户识破。

3. **两味药分开开**（实验 06 的教训）：`user_style` 治"这个人偏离人群的方向"，
   `model_tics` 治"模型自己的毛病"（对冲、每句都问回去、抢先认错）。
   实测：adapt 只开第一味时，对冲密度纹丝不动（8.28 → 7.29，人类 1.5）。
   混在一起给，模型会挑一条执行，所以两味药各自限额，且**不许同时超过 4 条**。

另一条实测教训（写进指令措辞，不是写进注释）：
   绝对措辞会被推到极值。"句末不用句号" → 标点率 0.005（人类带 0.022–0.050）；
   改成"很少用"才落在带内。所以这里所有指令都带**频率**，不带"绝不"。
"""

import re

from .metrics import measure

# ---------------------------------------------------------------- 用户风格
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
     "句末很少用句号——十句里最多一两句带标点，不是一句都不带。"),
    ("chunk_len_cv", "high",
     "长短差得很开，有的三五个字，有的整段",
     "长短交错：不要每条都差不多长，允许一条只有两三个字。"),
]

# ---------------------------------------------------------------- 模型毛病
# 阈值来自实验 06 的 A 臂（120 条裸提示输出）与人类带（baselines/chat.json）：
#   hedge_per_kchar  A=8.28   人类带 0.75–1.75
#   ask_back_rate    A=0.425  人：把话问回去是自然的，但每句都问是客服腔
#   apology_rate     A=0.025  真人很少在日常里抢先认错（用户 #11 的判词）
TICS = [
    ("hedge_per_kchar", 2.5,
     "对冲词（不过/其实/可能/仅供参考）密度偏高",
     "别每条都补一句缓冲：先把结论说完，需要转折时再说，不必每段都留后路。"),
    ("ask_back_rate", 0.30,
     "每句结尾都问回去（「你想聊点啥？」「你觉得呢？」）",
     "不是每次都要把话问回去：对方没问你就直接说完，允许一轮里不带问句。"),
    ("apology_rate", 0.05,
     "习惯抢先认错（「抱歉抱歉」「是我太着急了」）",
     "别抢着认错：只有在确实做错时才说抱歉，关系近时过度认错反而显得假。"),
]

# 削短时不许削掉的那半句——来自实验 06 弃权题 #29 #32
KEEP_WARM = ("短的是信息，不是态度：要削的是铺垫、解释、客套；"
             "关心、情绪、判断这些半句要留着。"
             "例：把「哦，还在忙啊？跑镜头是拍东西还是剪片呀？别太熬了」"
             "削成「镜头跑到哪一步了」就冷了——留一句关心再走。")

# ---------------------------------------------------------------- 交往规则（第三味药）
# 来源：实验 07 的八条弃权判词。他弃权的理由**没有一条是长度或标点**，全在行为层。
# 证据（原话节选）：
#   #6  「真人不会说后半句…如果关系好的话会在后半段损你一句」
#   #20 「起因就是为了好玩…真人碰见这问题只会损你」
#   #27 「每次拒绝发的都是这句话…他可以说『你疯了？我不想被砍头，别拉着我一块死』」
#   #38 「直接回一句『你平常也这么回夸人吗』这样更好」
#   #39 「都太过啰嗦…大可直接问到底说的啥」
#   #28 「没能察觉我情绪」
# 这一组不是结构，是关系：同样 12 个字，分布上看不出差别，人情上差一个世界。
# 所以它**不需要新的测量输入**（关系远近/情绪由模型从上下文判断），但它也不能
# 用指标验收——只能靠人看。上限 2 条，理由同 model_tics：多给会互相稀释。
RELATION = [
    ("roast",
     "关系近的时候该损一句，而不是解释",
     "关系近的时候，对方明显在玩、或者明显问了个不着调的问题，先损一句（「你疯了？」"
     "「你平常也这么夸人吗」），别当真去答；解释排在损后面，或者干脆不解释。"),
    ("praise_back",
     "被夸要回敬，别平铺直叙",
     "被夸的时候别只回「谢谢」「哈哈是吗」，回敬一句（「你平常也这么夸人吗」"
     "「就你会说」），带点得意或怀疑的劲儿。"),
    ("refuse_joke",
     "拒绝要开玩笑地拒，并且咬住他刚说的那句",
     "要拒绝的时候先抓住他上一句的话头，用玩笑顶回去（他聊政坛 →「你疯了？我不想被砍头，"
     "别拉着我一块死」），再补一句实话；不要用「抱歉，我不能…」这种模板。"),
    ("ask_blunt",
     "听不懂就直接问，不铺垫",
     "没听懂就一句「啥意思」「你说的是哪个」，不要先复述他的话再问——那些铺垫都是废话。"),
    ("feel_first",
     "对方有情绪时，第一句是态度，第二句才说事",
     "对方情绪明显时，先跟着乐、跟着急或跟着骂一句，再讲道理；别上来就分析。"),
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


def _starts(chunk, pats):
    head = chunk.strip()
    # 允许「哈哈，」「嗯，」这类语气词打头
    head = re.sub(r"^(哈+|嘿+|嗯|呃|啊|哦)[，,、\s]*", "", head)
    return any(head.startswith(p) for p in pats)


APOLOGY = ("抱歉", "对不起", "不好意思", "是我", "你说得对", "确实是我", "怪我")
ASK_TAIL = re.compile(r"(？|\?|吗|呢|吧)[\"”’）)]*[\s。！!…~～]*$")


def tic_rates(turns):
    """模型侧三项毛病的实测值（不依赖人群基线，只对比人类常数）。"""
    n = len(turns)
    if not n:
        return {"hedge_per_kchar": 0.0, "ask_back_rate": 0.0, "apology_rate": 0.0}
    rep = measure(turns)
    apol = sum(1 for t in turns if t and _starts(t[0], APOLOGY))
    ask = sum(1 for t in turns if t and ASK_TAIL.search(t[-1]))
    return {
        "hedge_per_kchar": rep.get("hedge_per_kchar", 0.0),
        "ask_back_rate": round(ask / n, 3),
        "apology_rate": round(apol / n, 3),
    }


def advise(turns, model_turns=None, max_user=2, max_tics=2, max_rel=2):
    """返回 {features, user_style, model_tics, relation, instructions, card}。

    turns       : 用户自己的历史（turn 级，list[list[str]]）
    model_turns : 助手自己的输出（可选）。不给就只用常数判"有这毛病"，
                  并把阈值写在指令里——**不假装测过**。
    max_rel     : 交往规则条数上限（默认 2；0 = 只开结构药）

    三味药各自限额、互不挤占：结构（这个人怎么说话）、毛病（模型自己的口癖）、
    关系（熟人之间怎么接话）。混在一起给，模型会挑一条执行。
    """
    rep = measure(turns)

    user_style = []
    for name, thresh, feature, cmd in RULES:
        t = "lowp" if name == "end_punct_rate" else thresh
        if _hit(name, t, rep):
            user_style.append({"metric": name, "value": rep[name],
                               "feature": feature, "instruction": cmd})
    user_style = user_style[:max_user]

    if model_turns:
        rates = tic_rates(model_turns)
        measured = True
    else:
        # 实验 06 A 臂的实测值，用作"没给模型语料时的默认画像"
        rates = {"hedge_per_kchar": 8.28, "ask_back_rate": 0.425, "apology_rate": 0.025}
        measured = False

    model_tics = []
    for name, thresh, feature, cmd in TICS:
        v = rates.get(name, 0.0)
        if name == "apology_rate":
            fire = v > thresh
        else:
            fire = v > thresh
        if measured and not fire:
            continue
        model_tics.append({"metric": name, "value": v, "feature": feature,
                           "instruction": cmd, "measured": measured})
    model_tics = model_tics[:max_tics]

    relation = [{"key": k, "feature": f, "instruction": c}
                for k, f, c in RELATION][:max(0, max_rel)]

    lines = []
    for p in user_style:
        lines.append("- 这个用户习惯%s（实测 %s = %s），所以输出用这种形态：%s"
                     % (p["feature"], p["metric"], p["value"], p["instruction"]))
    for p in model_tics:
        src = "实测 %s = %s" % (p["metric"], p["value"]) if p["measured"] else "模型默认如此"
        lines.append("- 这个模型的毛病是%s（%s），所以输出要避开：%s"
                     % (p["feature"], src, p["instruction"]))
    for p in relation:
        lines.append("- %s：%s" % (p["feature"], p["instruction"]))
    if lines:
        lines.append("- %s" % KEEP_WARM)

    return {
        "n_turns": rep["n_turns"],
        "features": user_style,
        "user_style": user_style,
        "model_tics": model_tics,
        "relation": relation,
        "instructions": lines,
        "card": "\n".join(lines) if lines else "（这个人的语料没触发任何形态规则，别硬套）",
    }
