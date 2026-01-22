import os
import re
import argparse
from datasets import load_dataset

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)  # windows illegal chars
    s = re.sub(r"\s+", "_", s)
    return s

def try_load_pacs():
    # 兼容不同HF上可能的dataset命名（脚本会自动尝试）
    candidates = [
        "flwrlabs/pacs",
        "pacs",
        "PACS",
        "kadirnar/pacs",
    ]
    last_err = None
    for name in candidates:
        try:
            ds = load_dataset(name)
            return name, ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load PACS from candidates={candidates}\nLast error: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help=r"e.g. D:\icip_osr\data\PACS")
    ap.add_argument("--max_n", type=int, default=0, help="0 means export all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds_name, ds = try_load_pacs()
    split = "train" if "train" in ds else list(ds.keys())[0]
    d = ds[split]

    # label names if available
    label_names = None
    try:
        label_names = d.features["label"].names
    except Exception:
        label_names = None

    n = len(d) if args.max_n <= 0 else min(len(d), args.max_n)
    print(f"[Info] Loaded dataset: {ds_name} split={split}, total={len(d)}")
    print(f"[Info] Exporting {n} samples -> {args.out_dir}")

    domains_seen = set()
    classes_seen = set()

    for i in range(n):
        ex = d[i]
        img = ex["image"]

        # 常见字段：domain/label；如果字段名不同，尽量兼容
        domain = ex.get("domain", ex.get("style", ex.get("source_domain", "unknown_domain")))
        label = ex.get("label", ex.get("y", ex.get("class", None)))
        if label is None:
            raise RuntimeError("Cannot find label field. Available keys: " + str(list(ex.keys())))

        domain = safe_name(domain)
        cls = safe_name(label_names[label] if label_names is not None and isinstance(label, int) else str(label))

        domains_seen.add(domain)
        classes_seen.add(cls)

        out_folder = os.path.join(args.out_dir, domain, cls)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{i:06d}.jpg")
        if not os.path.exists(out_path):
            img.convert("RGB").save(out_path, quality=95)

    print("[Done] Export finished.")
    print("[Done] Domains:", sorted(domains_seen))
    print("[Done] Classes:", sorted(classes_seen))

if __name__ == "__main__":
    main()
