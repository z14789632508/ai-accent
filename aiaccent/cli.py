# -*- coding: utf-8 -*-
import argparse
import json
import sys

from . import baseline as B
from . import io as I
from . import metrics as M
from . import mine as Mi
from . import prescribe as P


def _table(rows):
    out = []
    out.append("%-26s %10s %8s %8s %8s %7s  %s" % ("metric", "value", "p10", "p50", "p90", "dev", "verdict"))
    out.append("-" * 84)
    for name, v, lo, mid, hi, verdict, dev in rows:
        mark = {"in": "ok", "above": "HIGH", "below": "LOW"}[verdict]
        if abs(dev) >= B.SEVERE:
            mark += " (明显)"
        out.append("%-26s %10s %8s %8s %8s %7s  %s" % (name, v, lo, mid, hi, dev, mark))
    return "\n".join(out)


def cmd_baseline(args):
    regs = B.list_registers()
    if not regs:
        print("no baselines yet")
        return 1
    for r in regs:
        b = B.load(r)
        print("%-8s %-52s turns=%-7d chunks=%d" % (r, b.get("source", "")[:52], b["n_turns"], b["n_chunks"]))
    return 0


def cmd_audit(args):
    reg = args.register
    if not reg:
        print("--register is required (available: %s)" % ", ".join(B.list_registers()), file=sys.stderr)
        return 2
    turns = I.read_corpus(args.corpus)
    rep = M.measure(turns, sentence_bin=args.sentence_bin)
    base = B.load(reg)
    rows = B.compare(rep, base)

    if args.json:
        print(json.dumps({"report": rep, "baseline": reg, "rows": rows}, ensure_ascii=False, indent=1))
        return 0

    print("corpus   %s" % args.corpus)
    print("register %s   baseline: %s" % (reg, base.get("source", "")))
    print("turns %d  chunks %d" % (rep["n_turns"], rep["n_chunks"]))
    print()
    print(_table(rows))
    if args.prescribe:
        adv = P.advise(reg, rep, base, rows, provider=args.provider)
        print("\n== sampling ==")
        for k, v in adv["sampling"].items():
            print("  %-18s %s" % (k, v))
        for k in adv.get("ineffective", []):
            print("  %-18s (此后端不生效)" % k)
        if adv["shape_card"]:
            print("\n== shape card (贴进 system prompt) ==\n%s" % adv["shape_card"])
        print("\n== notes ==")
        for n in adv["notes"]:
            print("  - " + n)
    return 0


def cmd_diff(args):
    a = M.measure(I.read_corpus(args.a), sentence_bin=args.sentence_bin)
    b = M.measure(I.read_corpus(args.b), sentence_bin=args.sentence_bin)
    print("%-26s %12s %12s %12s" % ("metric", "A", "B", "delta"))
    print("-" * 66)
    for k in ("chunk_len_p50", "chunk_len_cv", "end_punct_rate", "turn_chunks_median",
              "turn_share_3", "turn_chunks_entropy", "within_turn_cv_median", "sentence_len_entropy"):
        if k in a and k in b:
            print("%-26s %12s %12s %12s" % (k, a[k], b[k], round(b[k] - a[k], 3)))
    return 0


def cmd_prescribe(args):
    reg = args.register
    rep = M.measure(I.read_corpus(args.corpus), sentence_bin=args.sentence_bin)
    base = B.load(reg)
    rows = B.compare(rep, base)
    adv = P.advise(reg, rep, base, rows, provider=args.provider)
    if args.json:
        print(json.dumps(adv, ensure_ascii=False, indent=1))
        return 0
    for k, v in adv["sampling"].items():
        print("%-18s %s" % (k, v))
    if adv["shape_card"]:
        print("\n" + adv["shape_card"] + "\n")
    for n in adv["notes"]:
        print("- " + n)
    return 0


def cmd_mine(args):
    base = B.load(args.register)
    turns = I.read_corpus(args.corpus)
    picked = Mi.mine(turns, base, k=args.k, max_chars=args.max_chars)
    if args.json:
        print(json.dumps(picked, ensure_ascii=False, indent=1))
    else:
        print(Mi.as_fewshot(picked))
    return 0


def build():
    p = argparse.ArgumentParser(prog="aiaccent", description="AI 口音体检表：不看词，看分布")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="体检一个语料")
    a.add_argument("corpus")
    a.add_argument("--register", required=False, help="chat / formal / ...（无默认值）")
    a.add_argument("--sentence-bin", type=int, default=10)
    a.add_argument("--prescribe", action="store_true", help="顺带给出采样参数与形状卡")
    a.add_argument("--provider", default="generic", choices=sorted(P.PROVIDER_SUPPORT),
                   help="按后端能力过滤处方（有些参数是空转的）")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_audit)

    d = sub.add_parser("diff", help="两个语料对比")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--sentence-bin", type=int, default=10)
    d.set_defaults(func=cmd_diff)

    s = sub.add_parser("prescribe", help="只出处方")
    s.add_argument("corpus")
    s.add_argument("--register", required=True)
    s.add_argument("--sentence-bin", type=int, default=10)
    s.add_argument("--provider", default="generic", choices=sorted(P.PROVIDER_SUPPORT))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_prescribe)

    m = sub.add_parser("mine", help="从真实语料挖 few-shot 样本")
    m.add_argument("corpus")
    m.add_argument("--register", required=True)
    m.add_argument("-k", type=int, default=20)
    m.add_argument("--max-chars", type=int, default=60)
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_mine)

    bl = sub.add_parser("baseline", help="列出已安装的基线")
    bl.set_defaults(func=cmd_baseline)

    return p


def main(argv=None):
    p = build()
    args = p.parse_args(argv)
    return args.func(args)
