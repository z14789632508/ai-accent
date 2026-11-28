"""实验 06 的盲测卷工具：生成 / 判分。

用法：
  python experiments/make_2afc.py build --a data/val/a.jsonl --c data/val/c.jsonl \
      --out data/val --n 40 --seed 7
  python experiments/make_2afc.py score --key data/val/key.json \
      --answers data/val/answers.json

输入 a.jsonl / c.jsonl 是 turn 级语料（每行一个 turn 的段列表，或 {"chunks": [...]}），
两个文件必须一一对应（第 i 行是同一条 prompt 的两组输出）。

answers.json 结构：{"1": {"q1": "1", "q2": "2", "q3": "1"}, ...}，数字是选项号。
q1 = 更想接着聊，q2 = 读着更顺，q3 = 更像在刻意学我说话（反向，选 C 侧视为失分）。
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aiaccent.io import read_corpus  # noqa: E402


def _text(turn):
    return "\n".join(turn)


def build(args):
    a = read_corpus(args.a)
    c = read_corpus(args.c)
    if len(a) != len(c):
        raise SystemExit("两条臂的 turn 数不一致：%d vs %d" % (len(a), len(c)))

    rng = random.Random(args.seed)
    idx = list(range(len(a)))
    rng.shuffle(idx)
    picked = sorted(idx[: args.n])

    os.makedirs(args.out, exist_ok=True)
    items, key = [], {}
    for no, i in enumerate(picked, 1):
        flip = rng.random() < 0.5
        left, right = (c[i], a[i]) if flip else (a[i], c[i])
        key[str(no)] = {"prompt_index": i, "c_side": "1" if flip else "2"}
        items.append((no, _text(left), _text(right)))

    path = os.path.join(args.out, "sheet.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 盲测卷（%d 组）\n\n" % len(items))
        fh.write("每题两问：q1 更想接着聊哪个；q2 哪个读着更顺；q3 哪个更像在刻意学我说话。\n")
        fh.write("填进 answers.json，形如 `{\"1\": {\"q1\": \"1\", \"q2\": \"2\", \"q3\": \"2\"}}`。\n\n")
        for no, left, right in items:
            fh.write("---\n\n## %d\n\n**1**\n\n%s\n\n**2**\n\n%s\n\n" % (no, left, right))
            fh.write("q1( ) q2( ) q3( )\n\n")

    kp = os.path.join(args.out, "key.json")
    with open(kp, "w", encoding="utf-8") as fh:
        json.dump(key, fh, ensure_ascii=False, indent=2)

    print("卷子：%s" % path)
    print("答案键：%s" % kp)
    print("组数：%d" % len(items))


def score(args):
    key = json.load(open(args.key, encoding="utf-8"))
    ans = json.load(open(args.answers, encoding="utf-8"))

    stat = {"q1": [0, 0], "q2": [0, 0], "q3": [0, 0]}  # [c 胜, 有效]
    for no, rec in key.items():
        a = ans.get(no)
        if not a:
            continue
        for q in ("q1", "q2", "q3"):
            pick = str(a.get(q, "")).strip()
            if pick not in ("1", "2"):
                continue
            c_won = (pick == rec["c_side"]) if q != "q3" else (pick != rec["c_side"])
            stat[q][1] += 1
            stat[q][0] += 1 if c_won else 0

    labels = {"q1": "更想接着聊", "q2": "读着更顺", "q3": "更不刻意迎合"}
    print("C 臂（adapt）相对 A 臂（原样）的胜率：")
    for q, (w, n) in stat.items():
        if n:
            print("  %s：%d/%d = %.1f%%" % (labels[q], w, n, 100.0 * w / n))
        else:
            print("  %s：无有效答案" % labels[q])

    if stat["q1"][1] >= 20:
        w, n = stat["q1"]
        ratio = w / n
        print()
        print("P1 判据（≥65%%，n≥20）：%s" % ("通过" if ratio >= 0.65 and n >= 20 else "未通过"))
        print("P3 还需要 D 臂的胜率一起算（C 胜率 − D 胜率 ≥ 15pp）。")


def main(argv=None):
    p = argparse.ArgumentParser(prog="make_2afc")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--a", required=True)
    b.add_argument("--c", required=True)
    b.add_argument("--out", default="data/val")
    b.add_argument("--n", type=int, default=40)
    b.add_argument("--seed", type=int, default=7)

    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--answers", required=True)

    args = p.parse_args(argv)
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
