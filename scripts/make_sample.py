# -*- coding: utf-8 -*-
"""生成 examples/sample.jsonl：**文案是手写的，分条方式由脚本按人类分布排**。

为什么不用真语料：真语料是别人私聊，不能进仓库。为什么不用随机打乱的字：
那样的 demo 一眼假，而且第一屏就没人看懂。所以：句子来自下面这张手写短语表，
"一轮拆几条 / 每条多长"由种子 + 参数决定，跑出来的指标落在 chat 基线带内。

用法：python scripts/make_sample.py [--seed 7]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aiaccent import metrics as M  # noqa: E402

# 手写短语：极短 / 中 / 长 三档，长度尽量拉开（人类 chunk_len_cv 高，靠的就是这个跨度）
TINY = ["嗯", "行", "去", "算", "也是", "累", "明白", "收到了", "谢了", "不急",
        "真的假的", "客气啥", "先看着呗", "八成开着", "放心", "哦对"]
MID = ["今天加班到九点才走", "饭还没吃，饿得前胸贴后背", "先随便吃点垫一下",
       "楼下那家面馆这个点还开着没有", "行那我过去一趟", "回来给你带一份",
       "不用不用，我自己煮包面就行", "包面也算一顿饭啊", "光有碳水没菜",
       "你这也太好养活了", "省钱嘛，到月底都这样", "周末那个展还去吗",
       "票我买好了", "几点开场你还记得吗", "我提前一小时叫你起来",
       "那不如叫我八点", "我想慢慢收拾一下", "八点我自己也起不来",
       "哈哈哈那算了，随缘", "地铁过去四十分钟", "其实不算远",
       "你今天话怎么这么少", "是不是累了", "脑子转不动了", "说话都费劲",
       "那你早点睡，别熬到一两点", "报表明天一早就要交", "又到月底了",
       "不过说真的", "这活干得越来越没劲了", "你干多久了，有三年了吧",
       "两年多，从实习生一路干上来", "那确实该动一动了", "现在这行情也不好说",
       "不投永远没机会", "我帮你问问我们那边", "最近好像在招人",
       "别专门麻烦你", "内推就是走个流程", "那我今晚把简历改一改",
       "改完发我看看", "帮你挑挑毛病", "白天实在没空", "下周再说也来得及",
       "回头请你吃饭", "你以前也帮过我不少次", "面记得吃，别放凉了",
       "到家跟我说一声", "我算着时间等你消息", "出了地铁再走十分钟就到",
       "我这边有事再找你", "别自己憋着", "刚才那个单子我看了", "数字对不上",
       "我这边算的是三万二", "怕是漏了一笔运费", "我重新核一遍",
       "十分钟给你结果", "核完直接发群里", "别单独发我", "顺便把上次那份补上",
       "上次那份还在你那儿吗", "我这台电脑重装过", "我存到共享盘了",
       "路径等下发你", "改完记得留一版备份", "别覆盖原文件，这事吃过大亏",
       "我习惯存两个版本，文件名带日期"]
LONG = [
    "我十点路过的时候灯还亮着，门口还坐着两个人在抽烟，看样子是通宵了",
    "之前那版写的是学校里做的项目，跟现在干的一点关系都没有，得整段重写",
    "上周开会的时候老板说下个月要调架构，我们这个组可能要拆成两半，谁走谁留还没定",
    "你先别急着改，等我明天把需求文档过一遍再发给你，免得到时候又要推翻重来",
    "那份报表我核到第三页发现口径不一样，去年的数是按老系统导的，今年换了新系统",
    "我记得你上次说过想做点自己的东西，现在还有这个想法吗，还是被工作磨没了",
]
TURN_SIZES = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6, 7]


def build(rnd, tiny_w, long_w):
    turns = []
    for _ in range(62):
        n = rnd.choice(TURN_SIZES)
        chunks = []
        for _ in range(n):
            r = rnd.random()
            if r < long_w:
                chunks.append(rnd.choice(LONG))
            elif r < long_w + tiny_w:
                chunks.append(rnd.choice(TINY))
            elif r < long_w + tiny_w + 0.18:
                chunks.append(rnd.choice(MID) + "，" + rnd.choice(MID))
            else:
                c = rnd.choice(MID)
                if rnd.random() < 0.05:
                    c += "。"
                chunks.append(c)
        turns.append(chunks)
    return turns


TARGET = {
    "chunk_len_p50": 12, "chunk_len_cv": 1.81, "end_punct_rate": 0.036,
    "turn_chunks_median": 2, "turn_share_3": 0.137,
    "turn_chunks_entropy": 3.02, "within_turn_cv_median": 0.512,
    "hedge_per_kchar": 1.25, "hedge_chunk_share": 0.023,
}


def score(r):
    s = 0.0
    for k, t in TARGET.items():
        v = r[k]
        s += abs(v - t) / (abs(t) or 1.0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "sample.jsonl"))
    a = ap.parse_args()

    best = None
    for sw in (0.12, 0.18, 0.24, 0.30, 0.36, 0.44):
        for lw in (0.02, 0.05, 0.08, 0.11, 0.14, 0.18):
            turns = build(random.Random(a.seed), sw, lw)
            sc = score(M.measure(turns))
            if best is None or sc < best[0]:
                best = (sc, sw, lw, turns)
    sc, sw, lw, turns = best
    with open(a.out, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    r = M.measure(turns)
    print("wrote %s  tiny=%.2f long=%.2f  err=%.3f" % (a.out, sw, lw, sc))
    for k in TARGET:
        print("  %-24s %s" % (k, r[k]))


if __name__ == "__main__":
    main()
