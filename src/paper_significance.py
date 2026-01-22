# paper_significance.py
# Generate significance tests + a ready-to-paste markdown report from paired_delta_vs_baseline.csv

import os, csv, math, argparse
from statistics import mean, stdev

def read_rows(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                return list(r)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read csv: {path}")

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def try_import_scipy():
    try:
        from scipy import stats  # type: ignore
        return stats
    except Exception:
        return None

def t_ci_95(mu, sd, n, stats):
    if n <= 1:
        return (mu, mu)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    return (mu - tcrit * se, mu + tcrit * se)

def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"

def one_sample_tests(xs, stats):
    # returns (t_p, wilcoxon_p)
    t_p = None
    w_p = None
    if stats is None or len(xs) <= 1:
        return (t_p, w_p)
    try:
        t_p = float(stats.ttest_1samp(xs, 0.0).pvalue)
    except Exception:
        t_p = None
    try:
        # Wilcoxon needs at least one non-zero
        if any(abs(v) > 0 for v in xs):
            w_p = float(stats.wilcoxon(xs).pvalue)
    except Exception:
        w_p = None
    return (t_p, w_p)

def summarize(rows, group_key=None):
    """
    rows: list[dict] with dKnownAcc, dOSRAcc, dAUROC, dFPR@95
    group_key: e.g. ("config",) or ("config","target_domain")
    """
    stats = try_import_scipy()
    metrics = ["dKnownAcc", "dOSRAcc", "dAUROC", "dFPR@95"]

    # group
    groups = {}
    for row in rows:
        k = tuple(row[g] for g in group_key) if group_key else ("ALL",)
        groups.setdefault(k, []).append(row)

    out = []
    for k, rs in sorted(groups.items()):
        for m in metrics:
            xs = [to_float(r.get(m)) for r in rs]
            xs = [v for v in xs if v is not None]
            n = len(xs)
            if n == 0:
                continue
            mu = mean(xs)
            sd = stdev(xs) if n > 1 else 0.0

            t_p, w_p = one_sample_tests(xs, stats)

            ci_low, ci_high = (mu, mu)
            if stats is not None and n > 1:
                ci_low, ci_high = t_ci_95(mu, sd, n, stats)

            pos = sum(1 for v in xs if v > 0)
            neg = sum(1 for v in xs if v < 0)
            zro = n - pos - neg

            out.append({
                "group": "|".join(k),
                "metric": m,
                "n": n,
                "mean": f"{mu:.6f}",
                "std": f"{sd:.6f}",
                "ci95_low": f"{ci_low:.6f}",
                "ci95_high": f"{ci_high:.6f}",
                "p_ttest": fmt_p(t_p),
                "p_wilcoxon": fmt_p(w_p),
                "pos/neg/zero": f"{pos}/{neg}/{zro}",
            })
    return out

def write_csv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired_csv", type=str, default="", help="paired_delta_vs_baseline.csv (default: auto find next to results_csv/out_dir)")
    ap.add_argument("--results_dir", type=str, default="", help="folder containing paired_delta_vs_baseline.csv")
    ap.add_argument("--out_dir", type=str, default="", help="default: same as paired_csv folder")
    args = ap.parse_args()

    paired_csv = args.paired_csv.strip()
    if not paired_csv:
        base = args.results_dir.strip()
        if not base:
            raise SystemExit("Provide --paired_csv or --results_dir")
        paired_csv = os.path.join(base, "paired_delta_vs_baseline.csv")

    if not os.path.exists(paired_csv):
        raise SystemExit(f"Not found: {paired_csv}")

    out_dir = args.out_dir.strip() or os.path.dirname(paired_csv)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_rows(paired_csv)

    overall = summarize(rows, group_key=("config",))
    by_target = summarize(rows, group_key=("config", "target_domain")) if "target_domain" in rows[0] else []

    overall_csv = os.path.join(out_dir, "significance_overall.csv")
    by_target_csv = os.path.join(out_dir, "significance_by_target.csv")
    header = ["group","metric","n","mean","std","ci95_low","ci95_high","p_ttest","p_wilcoxon","pos/neg/zero"]
    write_csv(overall_csv, overall, header)
    if by_target:
        write_csv(by_target_csv, by_target, header)

    # write a ready-to-paste markdown
    md_path = os.path.join(out_dir, "significance_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Significance report (from paired deltas)\n\n")
        f.write(f"- source: `{paired_csv}`\n\n")
        f.write("## Overall (grouped by config)\n\n")
        f.write("| group | metric | n | mean | std | 95% CI | p(t-test) | p(wilcoxon) | pos/neg/zero |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in overall:
            ci = f"[{r['ci95_low']}, {r['ci95_high']}]"
            f.write(f"| {r['group']} | {r['metric']} | {r['n']} | {r['mean']} | {r['std']} | {ci} | {r['p_ttest']} | {r['p_wilcoxon']} | {r['pos/neg/zero']} |\n")

        if by_target:
            f.write("\n## By target domain (grouped by config + target)\n\n")
            f.write("| group | metric | n | mean | std | 95% CI | p(t-test) | p(wilcoxon) | pos/neg/zero |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in by_target:
                ci = f"[{r['ci95_low']}, {r['ci95_high']}]"
                f.write(f"| {r['group']} | {r['metric']} | {r['n']} | {r['mean']} | {r['std']} | {ci} | {r['p_ttest']} | {r['p_wilcoxon']} | {r['pos/neg/zero']} |\n")

    print("[OK] wrote:")
    print(" -", overall_csv)
    if by_target:
        print(" -", by_target_csv)
    print(" -", md_path)

if __name__ == "__main__":
    main()
