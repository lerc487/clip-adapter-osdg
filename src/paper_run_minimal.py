# paper_run_minimal.py
# Minimal paper runner (FPR@95-centric) with streaming logs and auto tables.
# Works with:
#   - osr_clip_adapter.py  (your paper-ready version with --score_from auto/mix_alphas/calib_metric)
#   - paper_analyze_plus.py (the analyzer that writes tables)
#
# Usage example (in conda env):
#   python paper_run_minimal.py --data_root C:\icip_osr\data\PACS --num_splits 10 --seeds 0,1 --epochs 20 --early_stop 3

import os
import re
import csv
import sys
import json
import time
import math
import random
import hashlib
import argparse
import subprocess
from datetime import datetime
from itertools import combinations

# ----------- regex to parse osr_clip_adapter.py output -----------
RE_TARGET = re.compile(r"Target domain:\s*([A-Za-z0-9_]+)")
RE_KNOWN  = re.compile(r"Known Acc:\s*([0-9]*\.?[0-9]+)")
RE_OSR    = re.compile(r"OSR Acc:\s*([0-9]*\.?[0-9]+)")
RE_AUROC  = re.compile(r"AUROC:\s*([0-9]*\.?[0-9]+)")
RE_FPR95  = re.compile(r"FPR@95:\s*([0-9]*\.?[0-9]+)")

# Example calib line (paper-ready osr_clip_adapter.py):
# [Calib] score_source=mix  T_score=1.2  alpha=0.5  thr=...  maxF1=...  val_AUROC=...  val_FPR95=...
RE_CALIB = re.compile(
    r"\[Calib\].*?score_source=([a-z]+)\s+T_score=([0-9.]+)\s+alpha=([0-9A-Za-z\.\-]+)\s+thr=([0-9eE+\-\.]+)"
    r".*?(?:val_AUROC=([0-9.]+))?.*?(?:val_FPR95=([0-9.]+))?",
    re.IGNORECASE,
)

IMG_DOMAINS_EXPECTED = ("art_painting", "cartoon", "photo", "sketch")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_domains(data_root: str):
    ds = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    ds.sort()
    return ds


def list_classes_in_domain(data_root: str, domain: str):
    p = os.path.join(data_root, domain)
    cs = [c for c in os.listdir(p) if os.path.isdir(os.path.join(p, c))]
    cs.sort()
    return cs


def common_classes_across_domains(data_root: str, domains):
    sets = []
    for d in domains:
        sets.append(set(list_classes_in_domain(data_root, d)))
    common = sorted(list(set.intersection(*sets))) if sets else []
    return common


def stable_run_id(payload: dict):
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def read_existing_run_ids(csv_path: str):
    if not os.path.exists(csv_path):
        return set()
    ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = row.get("run_id", "")
            if rid:
                ids.add(rid)
    return ids


def write_csv_row(csv_path: str, fieldnames, row: dict):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def pick_val_domain(domains, target, forced=""):
    forced = (forced or "").strip()
    if forced:
        if forced == target:
            raise ValueError(f"--val_domain must be != target (got {forced} for target {target})")
        return forced
    # deterministic next-domain
    i = domains.index(target)
    return domains[(i + 1) % len(domains)]


def parse_result(text: str):
    mT = RE_TARGET.search(text)
    mK = RE_KNOWN.search(text)
    mO = RE_OSR.search(text)
    mA = RE_AUROC.search(text)
    mF = RE_FPR95.search(text)
    if not (mT and mK and mO and mA and mF):
        return None

    calib = {"score_source":"", "T_score":"", "alpha":"", "thr":"", "val_auroc":"", "val_fpr95":""}
    mc = RE_CALIB.search(text)
    if mc:
        calib["score_source"] = mc.group(1) or ""
        calib["T_score"] = mc.group(2) or ""
        calib["alpha"] = mc.group(3) or ""
        calib["thr"] = mc.group(4) or ""
        calib["val_auroc"] = (mc.group(5) or "")
        calib["val_fpr95"] = (mc.group(6) or "")

    return {
        "target_domain": mT.group(1),
        "known_acc": float(mK.group(1)),
        "osr_acc": float(mO.group(1)),
        "auroc": float(mA.group(1)),
        "fpr95": float(mF.group(1)),
        **calib,
    }


def stream_run(cmd_list, log_file):
    """
    Run subprocess with streaming stdout to both console and log file.
    Returns (returncode, full_output_text).
    """
    out_lines = []
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            out_lines.append(line)
        proc.wait()
    return proc.returncode, "".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--script", type=str, default="osr_clip_adapter.py", help="osr runner script path")
    ap.add_argument("--out_dir", type=str, default="", help="default: results/<timestamp>")
    ap.add_argument("--seeds", type=str, default="0,1", help="comma seeds, e.g. 0,1")
    ap.add_argument("--num_splits", type=int, default=10, help="how many unknown splits to sample (10 recommended). 0=all (can be slow)")
    ap.add_argument("--unknown_k", type=int, default=2)
    ap.add_argument("--targets", type=str, default="", help="comma list; empty => all domains")
    ap.add_argument("--val_domain", type=str, default="", help="force val domain; empty => auto per target (!=target)")
    ap.add_argument("--python_exe", type=str, default="", help="python executable; empty => current sys.executable")

    # speed / stability
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows建议0避免卡死)")
    ap.add_argument("--amp", action="store_true", default=True)

    # training knobs (keep small for speed)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--early_stop", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--hidden_ratio", type=float, default=0.0625)
    ap.add_argument("--max_scale", type=float, default=0.05)
    ap.add_argument("--cos_reg", type=float, default=0.2)

    # paper knobs (FPR@95)
    ap.add_argument("--calib_metric", type=str, default="fpr95", choices=["fpr95", "auroc"])
    ap.add_argument("--temp_candidates", type=str, default="0.2,0.3,0.4,0.5,0.6,0.8,1,1.2,1.5,2,3,4,5")
    ap.add_argument("--mix_alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")

    # analyzer
    ap.add_argument("--analyze", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    data_root = args.data_root.strip().strip('"')
    if not os.path.exists(data_root):
        raise SystemExit(f"[ERROR] data_root not found: {data_root}")

    py = args.python_exe.strip() or sys.executable
    script = args.script

    domains = list_domains(data_root)
    if not domains:
        raise SystemExit(f"[ERROR] no domains under: {data_root}")

    # sanity: PACS expected domains
    # (not strict; just helpful)
    if all(d in domains for d in IMG_DOMAINS_EXPECTED):
        pass

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else domains
    for t in targets:
        if t not in domains:
            raise SystemExit(f"[ERROR] unknown target '{t}', available: {domains}")

    common = common_classes_across_domains(data_root, domains)
    if len(common) < args.unknown_k + 1:
        raise SystemExit(f"[ERROR] too few common classes across domains: {common}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    # select unknown splits
    all_splits = list(combinations(common, args.unknown_k))
    rnd = random.Random(12345)
    rnd.shuffle(all_splits)
    if args.num_splits == 0:
        splits = all_splits
    else:
        splits = all_splits[: max(1, args.num_splits)]

    # outputs
    out_dir = args.out_dir.strip() or os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(out_dir)
    logs_dir = os.path.join(out_dir, "logs")
    ensure_dir(logs_dir)
    csv_path = os.path.join(out_dir, "results.csv")
    existing_ids = read_existing_run_ids(csv_path)

    prompt_ensemble = "a photo of a {}|a sketch of a {}|a cartoon of a {}|a painting of a {}"

    # ----------- Minimal configs (paper-friendly + much faster) -----------
    # 1) Fair baseline: zeroshot + held-out calibration tuned to MIN FPR@95
    # 2) Adapter scoring ablation: adapter (shows the problem / control)
    # 3) OURS: adapter classification + unknown score auto {zs/ad/mix(alphas)} + temp tuned on held-out val to MIN FPR@95
    def base_train_args(val_t: str):
        return [
            "--val_domain", val_t,
            "--epochs", str(args.epochs),
            "--early_stop", str(args.early_stop),
            "--early_stop_metric", "auroc",
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--hidden_ratio", str(args.hidden_ratio),
            "--max_scale", str(args.max_scale),
            "--cos_reg", str(args.cos_reg),
            "--templates", prompt_ensemble,
            "--num_workers", str(args.num_workers),
        ] + (["--amp"] if args.amp else [])

    def calib_args():
        return [
            "--tune_temp",
            "--temp_candidates", args.temp_candidates,
            "--calib_metric", args.calib_metric,
            "--mix_alphas", args.mix_alphas,
        ]

    configs = [
        {
            "name": "zeroshot_msp_promptens_calibFPR95",
            "extra_args_factory": lambda val_t: [
                "--baseline_zeroshot",
                "--score_mode", "msp",
                "--templates", prompt_ensemble,
                "--val_domain", val_t,
                "--score_from", "zeroshot",
                "--num_workers", str(args.num_workers),
            ] + calib_args() + (["--amp"] if args.amp else []),
        },
        {
            "name": "adapter_msp_promptens",
            "extra_args_factory": lambda val_t: base_train_args(val_t) + [
                "--score_mode", "msp",
                "--score_from", "adapter",
            ],
        },
        {
            "name": "adapter_msp_promptens_OURS_autoMixTemp_FPR95",
            "extra_args_factory": lambda val_t: base_train_args(val_t) + [
                "--score_mode", "msp",
                "--score_from", "auto",
            ] + calib_args(),
        },
    ]

    fieldnames = [
        "run_id", "timestamp",
        "config", "target_domain", "val_domain",
        "unknown_classes", "seed",
        "known_acc", "osr_acc", "auroc", "fpr95",
        "score_source", "T_score", "alpha", "thr", "val_auroc", "val_fpr95",
        "cmd", "log_file", "status", "error"
    ]

    total_jobs = len(targets) * len(splits) * len(seeds) * len(configs)
    print(f"[Info] python     = {py}")
    print(f"[Info] script     = {script}")
    print(f"[Info] data_root   = {os.path.abspath(data_root)}")
    print(f"[Info] out_dir     = {os.path.abspath(out_dir)}")
    print(f"[Info] domains     = {domains}")
    print(f"[Info] targets     = {targets}")
    print(f"[Info] seeds       = {seeds}")
    print(f"[Info] common_cls   = {common}")
    print(f"[Info] splits_used  = {len(splits)} (num_splits={args.num_splits})  unknown_k={args.unknown_k}")
    print(f"[Info] configs      = {[c['name'] for c in configs]}")
    print(f"[Info] epochs={args.epochs} early_stop={args.early_stop} num_workers={args.num_workers} amp={args.amp}")
    print(f"[Info] calib_metric={args.calib_metric} temps={args.temp_candidates} mix_alphas={args.mix_alphas}")
    print(f"[Info] total_jobs   = {total_jobs}")
    print("")

    done = 0
    for target in targets:
        val_t = pick_val_domain(domains, target, args.val_domain)
        for split in splits:
            unknown_str = ",".join(split)
            for seed in seeds:
                for cfg in configs:
                    extra = cfg["extra_args_factory"](val_t)

                    payload = {
                        "config": cfg["name"],
                        "target_domain": target,
                        "val_domain": val_t,
                        "unknown_classes": unknown_str,
                        "seed": seed,
                        "extra_args": extra,
                    }
                    run_id = stable_run_id(payload)
                    done += 1

                    if run_id in existing_ids:
                        print(f"[{done}/{total_jobs}] SKIP (already) {cfg['name']} target={target} val={val_t} unknown={unknown_str} seed={seed}")
                        continue

                    log_file = os.path.join(
                        logs_dir,
                        f"{run_id}_{cfg['name']}_{target}_val{val_t}_u{unknown_str}_s{seed}.txt"
                    )
                    cmd_list = [
                        py, script,
                        "--data_root", data_root,
                        "--target_domain", target,
                        "--unknown_classes", unknown_str,
                        "--seed", str(seed),
                    ] + extra

                    print(f"\n[{done}/{total_jobs}] {cfg['name']}  target={target}  val={val_t}  unknown={unknown_str}  seed={seed}")
                    print("CMD:", " ".join([f'"{x}"' if (" " in x or "|" in x) else x for x in cmd_list]))

                    if args.dry_run:
                        row = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "config": cfg["name"],
                            "target_domain": target,
                            "val_domain": val_t,
                            "unknown_classes": unknown_str,
                            "seed": seed,
                            "known_acc": "", "osr_acc": "", "auroc": "", "fpr95": "",
                            "score_source":"", "T_score":"", "alpha":"", "thr":"", "val_auroc":"", "val_fpr95":"",
                            "cmd": " ".join(cmd_list),
                            "log_file": log_file,
                            "status": "DRY_RUN",
                            "error": "",
                        }
                        write_csv_row(csv_path, fieldnames, row)
                        existing_ids.add(run_id)
                        continue

                    t0 = time.time()
                    try:
                        rc, out = stream_run(cmd_list, log_file)
                        parsed = parse_result(out)
                        if rc != 0:
                            status = "FAIL"
                            err = f"returncode={rc}"
                        elif parsed is None:
                            status = "PARSE_FAIL"
                            err = "could not parse metrics"
                        else:
                            status = "OK"
                            err = ""

                        row = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "config": cfg["name"],
                            "target_domain": target,
                            "val_domain": val_t,
                            "unknown_classes": unknown_str,
                            "seed": seed,
                            "known_acc": ("" if parsed is None else f"{parsed['known_acc']:.6f}"),
                            "osr_acc":   ("" if parsed is None else f"{parsed['osr_acc']:.6f}"),
                            "auroc":     ("" if parsed is None else f"{parsed['auroc']:.6f}"),
                            "fpr95":     ("" if parsed is None else f"{parsed['fpr95']:.6f}"),
                            "score_source": ("" if parsed is None else parsed.get("score_source","")),
                            "T_score":      ("" if parsed is None else parsed.get("T_score","")),
                            "alpha":        ("" if parsed is None else parsed.get("alpha","")),
                            "thr":          ("" if parsed is None else parsed.get("thr","")),
                            "val_auroc":    ("" if parsed is None else parsed.get("val_auroc","")),
                            "val_fpr95":    ("" if parsed is None else parsed.get("val_fpr95","")),
                            "cmd": " ".join(cmd_list),
                            "log_file": log_file,
                            "status": status,
                            "error": err,
                        }
                        write_csv_row(csv_path, fieldnames, row)
                        existing_ids.add(run_id)

                        dt = time.time() - t0
                        if status == "OK":
                            print(f"  -> OK in {dt:.1f}s  Known={row['known_acc']}  AUROC={row['auroc']}  FPR95={row['fpr95']}  (calib_src={row['score_source']},T={row['T_score']},alpha={row['alpha']})")
                        else:
                            print(f"  -> {status} in {dt:.1f}s  log={log_file}")

                    except KeyboardInterrupt:
                        print("\n[Interrupted] You can resume later; completed runs are skipped automatically.")
                        raise
                    except Exception as e:
                        row = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "config": cfg["name"],
                            "target_domain": target,
                            "val_domain": val_t,
                            "unknown_classes": unknown_str,
                            "seed": seed,
                            "known_acc": "", "osr_acc": "", "auroc": "", "fpr95": "",
                            "score_source":"", "T_score":"", "alpha":"", "thr":"", "val_auroc":"", "val_fpr95":"",
                            "cmd": " ".join(cmd_list),
                            "log_file": log_file,
                            "status": "EXCEPTION",
                            "error": repr(e),
                        }
                        write_csv_row(csv_path, fieldnames, row)
                        existing_ids.add(run_id)
                        print("  -> EXCEPTION:", e)

    print("\n[Done] results saved to:", os.path.abspath(csv_path))

    if args.analyze and (not args.dry_run):
        analyzer = os.path.join(os.path.dirname(__file__), "paper_analyze_plus.py")
        if os.path.exists(analyzer):
            print("[Analyze] Running paper_analyze_plus.py ...")
            subprocess.run([py, analyzer, "--results_csv", csv_path, "--std", "sample"], check=False)
            print("[Analyze] Outputs are in:", os.path.abspath(out_dir))
        else:
            print("[Analyze] Skipped: paper_analyze_plus.py not found next to this runner.")


if __name__ == "__main__":
    main()
