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
    prof = None
    if args.profile:
        from . import profile as PF
        prof = PF.load(args.profile)
        base = prof
        label = "profile:%s" % args.profile
    elif reg:
        base = B.load(reg)
        label = reg
    else:
        print("--register or --profile is required (baselines: %s / profiles: %s)"
              % (", ".join(B.list_registers()), ", ".join(_profiles())), file=sys.stderr)
        return 2
    turns = I.read_corpus(args.corpus)
    rep = M.measure(turns, sentence_bin=args.sentence_bin)
    rows = B.compare(rep, base, experimental=args.experimental)

    ci = None
    if args.ci:
        ci = M.bootstrap_ci(turns, [r["metric"] for r in rows],
                            reps=args.ci_reps, sentence_bin=args.sentence_bin)

    fit = None
    if prof is not None:
        from . import profile as PF
        fit = PF.fit(turns, prof)

    if args.json:
        print(json.dumps({"report": rep, "baseline": label, "rows": rows, "fit": fit,
                          "ci": {k: list(v) for k, v in (ci or {}).items()}},
                         ensure_ascii=False, indent=1))
        return 0

    print("corpus   %s" % args.corpus)
    print("%-8s %-11s band=%s" % ("profile" if prof else "register", label,
                                  base.get("band_method", "?")))
    print("baseline %s" % base.get("source", base.get("profile", "")))
    print("turns %d  chunks %d" % (rep["n_turns"], rep["n_chunks"]))
    print()
    print(_table(rows, ci))
    print("\npct = 该值在%s分布中的百分位；dev = 偏离该侧带宽的倍数（跨指标不可比）"
          % ("这个人自己的" if prof else "人类"))
    if fit and fit["total"]:
        print("\n== 指纹：这套口癖他有没有 ==")
        print("命中 %d/%d (%.0f%%)  判据：达到此人自身密度的 50%% 以上算在用"
              % (fit["hits"], fit["total"], 100 * fit["hit_rate"]))
        for r in fit["rows"]:
            print("  %s %-6s %5.2f/千字  (他 %5.2f, ×%s)"
                  % ("✓" if r["hit"] else "·", r["ngram"], r["per_kchar"],
                     r["person_per_kchar"], r["ratio"]))
    if args.prescribe:
        adv = P.advise(label, rep, base, rows, provider=args.provider)
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


def _profiles():
    try:
        from . import profile as PF
        return PF.list_profiles()
    except Exception:
        return []


def cmd_adapt(args):
    from . import adapt as A
    turns = I.read_corpus(args.corpus)
    adv = A.advise(turns, max_rules=args.max_rules)
    if args.json:
        print(json.dumps(adv, ensure_ascii=False, indent=1))
        return 0
    print("语料 %s  turns=%d" % (args.corpus, adv["n_turns"]))
    print("触发 %d 条形态规则（上限 %d，多了会被当噪声）\n" % (len(adv["features"]), args.max_rules))
    print(adv["card"])
    return 0


def cmd_profile(args):
    from . import profile as PF
    if args.action == "list":
        names = PF.list_profiles()
        if not names:
            print("no profiles yet")
            return 1
        for n in names:
            p = PF.load(n)
            print("%-12s turns=%-5d chunks=%-6d 指纹=%d条  %s"
                  % (n, p["n_turns"], p["n_chunks"], len(p.get("fingerprint") or []),
                     "、".join(x["ngram"] for x in (p.get("fingerprint") or [])[:6])))
        return 0

    turns = I.read_corpus(args.corpus)
    crowd = I.read_corpus(args.crowd) if args.crowd else None
    p, prof = PF.build(turns, args.name, crowd_turns=crowd,
                       replicates=args.replicates, resample_turns=args.resample_turns,
                       band_floor=args.band_floor, sentence_bin=args.sentence_bin)
    print("wrote %s  turns=%d chunks=%d 指纹=%d条"
          % (p, prof["n_turns"], prof["n_chunks"], len(prof["fingerprint"])))
    for e in prof["fingerprint"][:12]:
        print("  %-6s %5.2f/千字  人群 %s  超额 ×%s"
              % (e["ngram"], e["per_kchar"], e["crowd_per_kchar"], e["ratio"]))
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


def cmd_echo(args):
    from . import content as C
    pairs = C.read_pairs(args.pairs)
    if not pairs:
        print("没有可用的 pair（每行需要 {\"ctx\": …, \"reply\": …}）", file=sys.stderr)
        return 2
    st = C.echo_stats(pairs, n=args.n, worst=args.show)
    out = dict(st)
    if args.common:
        with open(args.common, encoding="utf-8") as f:
            common = set(json.load(f)["bigrams"])
        out["carry"] = C.carry_stats(pairs, common)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    print("pairs %d   判据：回复里逐字出现上下文的一段 %d 连字" % (st["n"], st["min_chars"]))
    print()
    print("复读率 %.1f%%  (%d/%d)" % (100 * st["echo_rate"], st["echo_n"], st["n"]))
    print("参考  人 3.8% / 机器 16.3%（QQ 私聊，见 experiments/05；样本量不同别直接比绝对值）")
    if "carry" in out and out["carry"]["n"]:
        c = out["carry"]
        print("\n承接率 %.3f  接住比例 %.3f  (n=%d，复读的对计 0)"
              % (c["carry_rate"], c["carry_hit"], c["n"]))
        print("注意  人聊天会换词，承接率低不等于没接住；这一项要配复读率一起读")
    for e in st.get("examples") or []:
        print("\n  ctx: %s\n  rep: %s" % (e["ctx"][:60], e["reply"][:60]))
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
    a.add_argument("--register", required=False, help="chat / chat_group / formal / ...")
    a.add_argument("--profile", required=False, help="个人档案名（baselines/profiles/<name>.json）")
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

    ad = sub.add_parser("adapt", help="从这个人的历史消息里读 3 条结构指令（不学用词）")
    ad.add_argument("corpus")
    ad.add_argument("--max-rules", type=int, default=3)
    ad.add_argument("--json", action="store_true")
    ad.set_defaults(func=cmd_adapt)

    pf = sub.add_parser("profile", help="个人档案：像不像某个具体的人")
    pf.add_argument("action", choices=["build", "list"])
    pf.add_argument("corpus", nargs="?", help="build 时的语料（turn 级 jsonl/txt）")
    pf.add_argument("--name", help="档案名，如 yg")
    pf.add_argument("--crowd", help="人群对照语料（算指纹的超额倍数用；不给就只按频次）")
    pf.add_argument("--replicates", type=int, default=1000)
    pf.add_argument("--resample-turns", type=int, default=300)
    pf.add_argument("--band-floor", type=float, default=0.4)
    pf.add_argument("--sentence-bin", type=int, default=10)
    pf.set_defaults(func=cmd_profile)

    ec = sub.add_parser("echo", help="内容层：这段回复是接住了还是复述了（吃成对的 ctx/reply）")
    ec.add_argument("pairs", help="jsonl，每行 {\"ctx\": 上下文, \"reply\": 回复}")
    ec.add_argument("-n", type=int, default=4, help="判复读的连字长度（默认 4）")
    ec.add_argument("--common", help="通用二元组表 json（给了才算承接率）")
    ec.add_argument("--show", type=int, default=0, help="打几条照搬的例子")
    ec.add_argument("--json", action="store_true")
    ec.set_defaults(func=cmd_echo)

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
