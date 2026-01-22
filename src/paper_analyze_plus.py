# paper_analyze_plus.py (paper-ready)
import os, csv, argparse
from collections import defaultdict
from statistics import mean, pstdev, stdev

METRICS = ["known_acc", "osr_acc", "auroc", "fpr95"]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mean_std(vals, use_sample_std: bool):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None, 0)
    if len(vals) == 1:
        return (vals[0], 0.0, 1)
    sd = stdev(vals) if use_sample_std else pstdev(vals)
    return (mean(vals), sd, len(vals))

def fmt(m, s):
    if m is None:
        return ""
    if s is None:
        return f"{m:.4f}"
    return f"{m:.4f}±{s:.4f}"

def norm_unknown(u: str) -> str:
    parts = [p.strip() for p in str(u).split(",") if p.strip()]
    parts.sort()
    return ",".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--baseline", type=str, default="")  # auto choose if empty
    ap.add_argument("--std", type=str, default="sample", choices=["sample", "pop"])
    args = ap.parse_args()

    results_csv = args.results_csv
    if not os.path.exists(results_csv):
        raise SystemExit(f"Not found: {results_csv}")

    out_dir = args.out_dir.strip() or os.path.dirname(results_csv)
    os.makedirs(out_dir, exist_ok=True)

    use_sample_std = (args.std == "sample")

    rows = []
    with open(results_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "OK":
                continue
            for k in METRICS:
                row[k] = to_float(row.get(k))
            row["seed"] = row.get("seed")
            row["unknown_norm"] = norm_unknown(row.get("unknown_classes", ""))
            rows.append(row)

    # group by (config, target)
    grp = defaultdict(list)
    configs = set()
    targets = set()
    for row in rows:
        key = (row["config"], row["target_domain"])
        grp[key].append(row)
        configs.add(row["config"])
        targets.add(row["target_domain"])

    configs = sorted(configs)
    targets = sorted(targets)

    # pick baseline
    baseline = args.baseline.strip()
    if not baseline:
        baseline = "zeroshot_msp_promptens" if "zeroshot_msp_promptens" in configs else (configs[0] if configs else "")
    if not baseline:
        raise SystemExit("No valid baseline config found in results.")

    # 1) main table
    table_path = os.path.join(out_dir, "main_table_mean_std.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "target_domain",
                    "KnownAcc(mean±std)", "OSRAcc(mean±std)", "AUROC(mean±std)", "FPR@95(mean±std)", "n_runs"])
        for cfg in configs:
            for tgt in targets:
                g = grp.get((cfg, tgt), [])
                mK, sK, n = mean_std([x["known_acc"] for x in g], use_sample_std)
                mO, sO, _ = mean_std([x["osr_acc"] for x in g], use_sample_std)
                mA, sA, _ = mean_std([x["auroc"] for x in g], use_sample_std)
                mF, sF, _ = mean_std([x["fpr95"] for x in g], use_sample_std)
                if n == 0:
                    continue
                w.writerow([cfg, tgt, fmt(mK, sK), fmt(mO, sO), fmt(mA, sA), fmt(mF, sF), n])

    # 2) missing matrix
    miss_path = os.path.join(out_dir, "missing_matrix.csv")
    with open(miss_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config"] + targets)
        for cfg in configs:
            row = [cfg]
            for tgt in targets:
                row.append("OK" if (cfg, tgt) in grp else "MISSING")
            w.writerow(row)

    # 3) best by target (min mean FPR, tie: max mean AUROC)
    best_path = os.path.join(out_dir, "best_by_target.csv")
    with open(best_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_domain", "best_config", "FPR@95_mean", "AUROC_mean", "KnownAcc_mean", "OSRAcc_mean", "n_runs"])
        for tgt in targets:
            cand = []
            for cfg in configs:
                g = grp.get((cfg, tgt), [])
                if not g:
                    continue
                mF, _, n = mean_std([x["fpr95"] for x in g], use_sample_std)
                mA, _, _ = mean_std([x["auroc"] for x in g], use_sample_std)
                mK, _, _ = mean_std([x["known_acc"] for x in g], use_sample_std)
                mO, _, _ = mean_std([x["osr_acc"] for x in g], use_sample_std)
                if mF is None or mA is None:
                    continue
                cand.append((cfg, mF, mA, mK, mO, n))
            if not cand:
                continue
            cand.sort(key=lambda x: (x[1], -x[2]))
            best = cand[0]
            w.writerow([tgt, best[0], f"{best[1]:.6f}", f"{best[2]:.6f}", f"{best[3]:.6f}", f"{best[4]:.6f}", best[5]])

    # 4) macro average on intersection targets
    common_targets = [t for t in targets if all((cfg, t) in grp for cfg in configs)]
    macro_path = os.path.join(out_dir, "macro_avg.csv")
    with open(macro_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "targets_used", "KnownAcc_mean", "OSRAcc_mean", "AUROC_mean", "FPR@95_mean"])
        for cfg in configs:
            vals = {k: [] for k in METRICS}
            for tgt in common_targets:
                g = grp[(cfg, tgt)]
                for k in METRICS:
                    m, _, _ = mean_std([x[k] for x in g], use_sample_std)
                    if m is not None:
                        vals[k].append(m)
            if not vals["auroc"]:
                continue
            w.writerow([
                cfg,
                "|".join(common_targets),
                f"{mean(vals['known_acc']):.6f}",
                f"{mean(vals['osr_acc']):.6f}",
                f"{mean(vals['auroc']):.6f}",
                f"{mean(vals['fpr95']):.6f}",
            ])

    # 5) delta vs baseline (per target)
    delta_path = os.path.join(out_dir, "delta_vs_baseline.csv")
    with open(delta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "target_domain", "dKnownAcc", "dOSRAcc", "dAUROC", "dFPR@95"])
        for cfg in configs:
            if cfg == baseline:
                continue
            for tgt in targets:
                gB = grp.get((baseline, tgt), [])
                gC = grp.get((cfg, tgt), [])
                if not gB or not gC:
                    continue
                mBk, _, _ = mean_std([x["known_acc"] for x in gB], use_sample_std)
                mBo, _, _ = mean_std([x["osr_acc"] for x in gB], use_sample_std)
                mBa, _, _ = mean_std([x["auroc"] for x in gB], use_sample_std)
                mBf, _, _ = mean_std([x["fpr95"] for x in gB], use_sample_std)

                mCk, _, _ = mean_std([x["known_acc"] for x in gC], use_sample_std)
                mCo, _, _ = mean_std([x["osr_acc"] for x in gC], use_sample_std)
                mCa, _, _ = mean_std([x["auroc"] for x in gC], use_sample_std)
                mCf, _, _ = mean_std([x["fpr95"] for x in gC], use_sample_std)

                w.writerow([cfg, tgt,
                            f"{(mCk-mBk):.6f}",
                            f"{(mCo-mBo):.6f}",
                            f"{(mCa-mBa):.6f}",
                            f"{(mCf-mBf):.6f}"])

    # 6) paired deltas vs baseline (same target/unknown/seed)
    paired_path = os.path.join(out_dir, "paired_delta_vs_baseline.csv")
    # index rows by key
    by_cfg = defaultdict(dict)
    for row in rows:
        key = (row["target_domain"], row["unknown_norm"], row["seed"])
        by_cfg[row["config"]][key] = row

    with open(paired_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "target_domain", "unknown_classes(norm)", "seed",
                    "dKnownAcc", "dOSRAcc", "dAUROC", "dFPR@95"])
        base_map = by_cfg.get(baseline, {})
        for cfg in configs:
            if cfg == baseline:
                continue
            cur_map = by_cfg.get(cfg, {})
            for key, brow in base_map.items():
                if key not in cur_map:
                    continue
                crow = cur_map[key]
                w.writerow([
                    cfg, key[0], key[1], key[2],
                    "" if (crow["known_acc"] is None or brow["known_acc"] is None) else f"{crow['known_acc']-brow['known_acc']:.6f}",
                    "" if (crow["osr_acc"] is None or brow["osr_acc"] is None) else f"{crow['osr_acc']-brow['osr_acc']:.6f}",
                    "" if (crow["auroc"] is None or brow["auroc"] is None) else f"{crow['auroc']-brow['auroc']:.6f}",
                    "" if (crow["fpr95"] is None or brow["fpr95"] is None) else f"{crow['fpr95']-brow['fpr95']:.6f}",
                ])

    # 7) paired delta summary (mean±std over all pairs)
    summ_path = os.path.join(out_dir, "paired_delta_summary.csv")
    # collect per config
    diffs = defaultdict(lambda: defaultdict(list))
    with open(paired_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cfg = row["config"]
            for m in ["dKnownAcc", "dOSRAcc", "dAUROC", "dFPR@95"]:
                v = to_float(row.get(m))
                if v is not None:
                    diffs[cfg][m].append(v)

    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "pairs", "dKnownAcc(mean±std)", "dOSRAcc(mean±std)", "dAUROC(mean±std)", "dFPR@95(mean±std)"])
        for cfg in configs:
            if cfg == baseline:
                continue
            pairs = len(diffs[cfg]["dAUROC"])
            def ms(key):
                m, s, _ = mean_std(diffs[cfg][key], use_sample_std)
                return fmt(m, s)
            w.writerow([cfg, pairs, ms("dKnownAcc"), ms("dOSRAcc"), ms("dAUROC"), ms("dFPR@95")])

    # 8) quick markdown summary (paper_tables.md)
    md_path = os.path.join(out_dir, "paper_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Paper tables (auto-generated)\n\n")
        f.write(f"- results_csv: `{os.path.abspath(results_csv)}`\n")
        f.write(f"- baseline: `{baseline}`\n\n")
        f.write("## Macro average\n\n")
        with open(macro_path, "r", encoding="utf-8") as mf:
            f.write("```text\n")
            f.write(mf.read())
            f.write("```\n\n")
        f.write("## Paired delta summary vs baseline\n\n")
        with open(summ_path, "r", encoding="utf-8") as sf:
            f.write("```text\n")
            f.write(sf.read())
            f.write("```\n\n")
        f.write("## Best by target (min mean FPR@95)\n\n")
        with open(best_path, "r", encoding="utf-8") as bf:
            f.write("```text\n")
            f.write(bf.read())
            f.write("```\n")

    print("[OK] Wrote:")
    print(" -", table_path)
    print(" -", best_path)
    print(" -", miss_path)
    print(" -", macro_path)
    print(" -", delta_path)
    print(" -", paired_path)
    print(" -", summ_path)
    print(" -", md_path)
    print("[Info] baseline =", baseline)
    print("[Info] std mode =", args.std)

if __name__ == "__main__":
    main()
