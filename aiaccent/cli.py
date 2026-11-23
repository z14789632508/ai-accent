# -*- coding: utf-8 -*-
import argparse
import json
import sys

from . import __version__
from . import baseline as B
from . import io as I
from . import metrics as M
from . import mine as Mi
from . import prescribe as P


def _table(rows, ci=None):
    out = []
    out.append("%-24s %9s %7s %7s %7s %7s %7s  %s"
               % ("metric", "value", "p10", "p50", "p90", "pct", "dev", "verdict"))
    out.append("-" * 86)
    for r in rows:
        mark = {"in": "ok", "above": "HIGH", "below": "LOW"}[r["verdict"]]
        if abs(r["dev"]) >= B.SEVERE:
            mark += " (明显)"
        pct = "%5.1f%%" % r["pct"]
        val = r["value"]
        if ci and r["metric"] in ci:
            lo, hi = ci[r["metric"]]
            val = "%s±%s" % (val, round((hi - lo) / 2, 2))
        out.append("%-24s %9s %7s %7s %7s %7s %7s  %s"
                   % (r["metric"], val, r["p10"], r["p50"], r["p90"], pct, r["dev"], mark))
    return "\n".join(out)


def cmd_baseline(args):
    regs = B.list_registers()
    if not regs:
        print("no baselines yet")
        return 1
    for r in regs:
        b = B.load(r)
        print("%-11s turns=%-6d chunks=%-7d band_method=%-16s %s"
              % (r, b["n_turns"], b["n_chunks"], b.get("band_method", "?"),
                 b.get("source", "")[:46]))
    return 0


def cmd_audit(args):
    reg = args.register
    if not reg:
        print("--register is required (available: %s)" % ", ".join(B.list_registers()),
              file=sys.stderr)
        return 2
    turns = I.read_corpus(args.corpus)
    rep = M.measure(turns, sentence_bin=args.sentence_bin)
    base = B.load(reg)
    rows = B.compare(rep, base, experimental=args.experimental)

    ci = None
    if args.ci:
        ci = M.bootstrap_ci(turns, [r["metric"] for r in rows],
                            reps=args.ci_reps, sentence_bin=args.sentence_bin)

    if args.json:
        print(json.dumps({"report": rep, "baseline": reg, "rows": rows,
                          "ci": {k: list(v) for k, v in (ci or {}).items()}},
                         ensure_ascii=False, indent=1))
        return 0

    print("corpus   %s" % args.corpus)
    print("register %-11s band=%s" % (reg, base.get("band_method", "?")))
    print("baseline %s" % base.get("source", ""))
    print("turns %d  chunks %d" % (rep["n_turns"], rep["n_chunks"]))
    print()
    print(_table(rows, ci))
    print("\npct = 该值在人类分布中的百分位；dev = 偏离该侧带宽的倍数（跨指标不可比）")
    if args.prescribe:
        adv = P.advise(reg, rep, base, rows, provider=args.provider)
        print("\n== sampling ==")
        for k, v in adv["sampling"].items():
            print("  %-18s %s" % (k, v))
        for k in adv.get("ineffective", []):
            print("  %-18s (此后端上不生效)" % k)
        if adv["shape_card"]:
            print("\n== shape card (贴进 system prompt) ==\n%s" % adv["shape_card"])
        print("\n== notes ==")
        for n in adv["notes"]:
            print("  - " + n)
    return 0


def cmd_diff(args):
    a = M.measure(I.read_corpus(args.a), sentence_bin=args.sentence_bin)
    b = M.measure(I.read_corpus(args.b), sentence_bin=args.sentence_bin)
    names = [r for r in B.CORE + B.EXPERIMENTAL if r in a and r in b]
    print("%-26s %12s %12s %12s" % ("metric", "A", "B", "delta"))
    print("-" * 66)
    for k in names:
        print("%-26s %12s %12s %12s" % (k, a[k], b[k], round(b[k] - a[k], 3)))
    return 0


def cmd_prescribe(args):
    rep = M.measure(I.read_corpus(args.corpus), sentence_bin=args.sentence_bin)
    base = B.load(args.register)
    rows = B.compare(rep, base, experimental=args.experimental)
    adv = P.advise(args.register, rep, base, rows, provider=args.provider)
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


def cmd_compare_provider(args):
    from . import providers as PR
    base = B.load(args.register)
    rows_out = []
    for spec in args.provider:
        name, out_path = PR.run(spec, args.prompts, args.out_dir,
                                max_tokens=args.max_tokens, temperature=args.temperature)
        turns = I.read_corpus(out_path)
        rep = M.measure(turns, sentence_bin=args.sentence_bin)
        rows = B.compare(rep, base, experimental=args.experimental)
        rows_out.append({"provider": name, "corpus": out_path, "report": rep, "rows": rows})

    if args.json:
        print(json.dumps(rows_out, ensure_ascii=False, indent=1))
        return 0

    print("prompts %s   register %s   baseline: %s"
          % (args.prompts, args.register, base.get("source", "")))
    for item in rows_out:
        print("\n== %s (%d turns) ==" % (item["provider"], item["report"]["n_turns"]))
        print(_table(item["rows"]))
    # 横向：谁离人更近
    print("\n== 离人类区间多远（只数核心指标，pct 越靠外越远）==")
    print("%-28s %10s %10s" % ("provider", "越界指标", "平均越界|pct-50|"))
    for item in rows_out:
        out = [r for r in item["rows"] if r["verdict"] != "in"]
        dist = sum(abs(r["pct"] - 50) for r in out) / len(out) if out else 0.0
        print("%-28s %10d %10.1f" % (item["provider"], len(out), dist))
    return 0


def build():
    p = argparse.ArgumentParser(prog="aiaccent", description="AI 口音体检表：不看词，看分布")
    p.add_argument("--version", action="version", version="aiaccent %s" % __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="体检一个语料")
    a.add_argument("corpus")
    a.add_argument("--register", required=False, help="chat / chat_group / formal / ...（无默认值）")
    a.add_argument("--sentence-bin", type=int, default=10)
    a.add_argument("--experimental", action="store_true", help="连实验性指标一起出（默认隐藏）")
    a.add_argument("--ci", action="store_true", help="对被测语料做 bootstrap 95% CI")
    a.add_argument("--ci-reps", type=int, default=400)
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
    s.add_argument("--experimental", action="store_true")
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

    cp = sub.add_parser("compare-provider", help="同一批 prompt 打多个模型，比谁更像人")
    cp.add_argument("prompts", help="每行一个 prompt 的 jsonl（{p} 或纯文本行）")
    cp.add_argument("--provider", action="append", required=True,
                    help="vendor:model，可重复，如 deepseek:deepseek-flash")
    cp.add_argument("--register", required=True)
    cp.add_argument("--out-dir", default="data/compare")
    cp.add_argument("--max-tokens", type=int, default=200)
    cp.add_argument("--temperature", type=float, default=1.0)
    cp.add_argument("--sentence-bin", type=int, default=10)
    cp.add_argument("--experimental", action="store_true")
    cp.add_argument("--json", action="store_true")
    cp.set_defaults(func=cmd_compare_provider)

    return p


def main(argv=None):
    p = build()
    args = p.parse_args(argv)
    return args.func(args)
