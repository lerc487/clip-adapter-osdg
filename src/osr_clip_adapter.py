import os, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip
from sklearn.metrics import roc_auc_score

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_domains(data_root: str) -> List[str]:
    ds = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    ds.sort()
    return ds


def list_classes(data_root: str, domain: str) -> List[str]:
    p = os.path.join(data_root, domain)
    cs = [c for c in os.listdir(p) if os.path.isdir(os.path.join(p, c))]
    cs.sort()
    return cs


def common_classes_across_domains(data_root: str, domains: List[str]) -> List[str]:
    sets = []
    for d in domains:
        sets.append(set(list_classes(data_root, d)))
    if not sets:
        return []
    return sorted(list(set.intersection(*sets)))


def collect_samples(data_root: str, domain: str, class_names: List[str]) -> List[Tuple[str, str]]:
    """Return (path, class_name) samples."""
    samples = []
    dom_dir = os.path.join(data_root, domain)
    for cn in class_names:
        cdir = os.path.join(dom_dir, cn)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith(IMG_EXTS):
                samples.append((os.path.join(cdir, fn), cn))
    return samples


class MultiDomainFolder(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


class Adapter(nn.Module):
    """
    Minimal CLIP adapter:
      feat' = normalize(feat + clamp(alpha)*MLP(feat))
    - hidden_ratio controls MLP width
    - alpha is a learnable scalar (init alpha_init), clamped to [-max_scale, max_scale]
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.5, max_scale: float = 0.1, alpha_init: float = 1.0):
        super().__init__()
        h = max(1, int(dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.max_scale = float(max_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.alpha, -self.max_scale, self.max_scale)
        return x + a * self.net(x)


@torch.no_grad()
def build_text_features(model, tokenizer, classnames: List[str], device, templates: List[str]) -> torch.Tensor:
    """
    Prompt ensemble: for each template, encode text, then average and normalize.
    Returns (C, D) normalized text features.
    """
    feats = []
    for tpl in templates:
        texts = [tpl.format(c.replace("_", " ")) for c in classnames]
        tok = tokenizer(texts).to(device)
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
        feats.append(t)
    t = torch.stack(feats, dim=0).mean(dim=0)
    t = t / t.norm(dim=-1, keepdim=True)
    return t


def score_unknown_from_logits(logits: np.ndarray, T: float, score_mode: str) -> np.ndarray:
    """
    Return unknown score, higher => more unknown.
    MSP: 1 - max softmax(logits/T)
    Energy: -T * logsumexp(logits/T)  (higher/less negative => more unknown)
    """
    x = logits / float(T)
    if score_mode == "msp":
        x = x - x.max(axis=1, keepdims=True)
        p = np.exp(x)
        p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        return 1.0 - p.max(axis=1)
    if score_mode == "energy":
        m = x.max(axis=1, keepdims=True)
        lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True) + 1e-12)
        energy = (-float(T) * lse.squeeze(1))
        return energy
    raise ValueError(f"Unknown score_mode: {score_mode}")


def pick_threshold_maxf1(known_scores: np.ndarray, unknown_scores: np.ndarray) -> Tuple[float, float]:
    """
    Pick threshold for "unknown if score >= thr" maximizing F1 for unknown (positive class).
    """
    all_scores = np.concatenate([known_scores, unknown_scores], axis=0)
    y = np.concatenate([np.zeros_like(known_scores, dtype=np.int64),
                        np.ones_like(unknown_scores, dtype=np.int64)], axis=0)

    order = np.argsort(-all_scores)
    s = all_scores[order]
    y = y[order]

    P = int((y == 1).sum())
    if P == 0:
        return float(np.max(all_scores) + 1e-6), 0.0

    tp = 0
    fp = 0
    best_f1 = -1.0
    best_thr = float(s[0])

    for i in range(len(s)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        fn = P - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(s[i])
    return best_thr, float(best_f1)


def fpr_at_95_tpr(scores: np.ndarray, y_unknown: np.ndarray) -> float:
    """
    y_unknown: 1 for unknown, 0 for known.
    scores: higher => more unknown.
    """
    order = np.argsort(-scores)
    y = y_unknown[order]
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P <= 0 or N <= 0:
        return 1.0
    tp = 0.0
    fp = 0.0
    target_tp = 0.95 * P
    for yi in y:
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0
        if tp >= target_tp:
            return float(fp / (N + 1e-12))
    return 1.0


@dataclass
class OSRResult:
    known_acc: float
    osr_acc: float
    auroc: float
    fpr95: float


@torch.no_grad()
def forward_logits(model, adapter: Optional[nn.Module], text_feat: torch.Tensor, images: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logits_zs, logits_ad) on GPU.
    If adapter is None, logits_ad == logits_zs (for unified downstream code).
    """
    images = images.to(device, non_blocking=True)
    feat = model.encode_image(images)
    feat = feat / feat.norm(dim=-1, keepdim=True)

    # logit scale
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = torch.tensor(1.0, device=device)

    logits_zs = (feat @ text_feat.T) * logit_scale

    if adapter is None:
        return logits_zs, logits_zs

    feat_ad = adapter(feat)
    feat_ad = feat_ad / feat_ad.norm(dim=-1, keepdim=True)
    logits_ad = (feat_ad @ text_feat.T) * logit_scale
    return logits_zs, logits_ad


def collect_logits_on_loader(model, adapter, text_feat, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (logits_zs, logits_ad, labels) as numpy arrays.
    Labels: known class id or -1 for unknown.
    """
    all_zs, all_ad, all_y = [], [], []
    for images, y in loader:
        lz, la = forward_logits(model, adapter, text_feat, images, device)
        all_zs.append(lz.detach().cpu().numpy())
        all_ad.append(la.detach().cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_zs, 0), np.concatenate(all_ad, 0), np.concatenate(all_y, 0)


def _eval_val_metrics_for_candidate(
    score_mode: str,
    T: float,
    k_logits: np.ndarray,
    u_logits: np.ndarray,
) -> Dict[str, float]:
    sk = score_unknown_from_logits(k_logits, T, score_mode)
    su = score_unknown_from_logits(u_logits, T, score_mode)
    y = np.concatenate([np.zeros_like(sk, np.int64), np.ones_like(su, np.int64)], 0)
    s = np.concatenate([sk, su], 0)
    auc = float(roc_auc_score(y, s))
    fpr95 = float(fpr_at_95_tpr(s, y))
    return {"auroc": auc, "fpr95": fpr95}


def choose_score_source_and_temp(
    score_from: str,
    score_mode: str,
    temps: List[float],
    val_known_logits_zs: np.ndarray,
    val_known_logits_ad: np.ndarray,
    val_unknown_logits_zs: np.ndarray,
    val_unknown_logits_ad: np.ndarray,
    tune_temp: bool,
    mix_alpha: float,
    mix_alphas: List[float],
    calib_metric: str,  # "auroc" or "fpr95"
) -> Dict[str, Any]:
    """
    Decide which logits to use for unknown scoring and pick temperature T on val.

    Returns dict:
      {
        "score_source": "adapter"|"zeroshot"|"mix",
        "T_score": float,
        "alpha": float or None,
        "val_auroc": float,
        "val_fpr95": float
      }

    If score_from=auto, candidates include:
      - zeroshot
      - adapter
      - mix with alpha in mix_alphas
    Objective:
      - calib_metric="fpr95": minimize val FPR@95, tie-break max AUROC
      - calib_metric="auroc": maximize val AUROC, tie-break min FPR@95
    """
    # If no unknown in val, fall back
    if val_unknown_logits_zs.size == 0:
        return {"score_source": ("adapter" if score_from in ("auto", "adapter", "mix") else "zeroshot"),
                "T_score": 1.0, "alpha": (mix_alpha if score_from == "mix" else None),
                "val_auroc": float("nan"), "val_fpr95": float("nan")}

    cand_T = temps if tune_temp else [1.0]

    def get_logits(src: str, alpha: Optional[float]):
        if src == "zeroshot":
            return val_known_logits_zs, val_unknown_logits_zs
        if src == "adapter":
            return val_known_logits_ad, val_unknown_logits_ad
        if src == "mix":
            a = float(alpha if alpha is not None else mix_alpha)
            k = a * val_known_logits_ad + (1.0 - a) * val_known_logits_zs
            u = a * val_unknown_logits_ad + (1.0 - a) * val_unknown_logits_zs
            return k, u
        raise ValueError(src)

    def better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Return True if a better than b."""
        if calib_metric == "fpr95":
            # smaller fpr95 better
            if a["val_fpr95"] < b["val_fpr95"] - 1e-12:
                return True
            if abs(a["val_fpr95"] - b["val_fpr95"]) <= 1e-12 and a["val_auroc"] > b["val_auroc"] + 1e-12:
                return True
            return False
        else:
            # larger auroc better
            if a["val_auroc"] > b["val_auroc"] + 1e-12:
                return True
            if abs(a["val_auroc"] - b["val_auroc"]) <= 1e-12 and a["val_fpr95"] < b["val_fpr95"] - 1e-12:
                return True
            return False

    # Build candidates
    candidates: List[Tuple[str, Optional[float]]] = []
    if score_from == "auto":
        candidates.append(("zeroshot", None))
        candidates.append(("adapter", None))
        for a in mix_alphas:
            candidates.append(("mix", float(a)))
    elif score_from == "mix":
        candidates.append(("mix", float(mix_alpha)))
    elif score_from in ("adapter", "zeroshot"):
        candidates.append((score_from, None))
    else:
        raise ValueError(f"score_from={score_from}")

    best = {"score_source": candidates[0][0], "T_score": 1.0, "alpha": candidates[0][1],
            "val_auroc": float("inf") if calib_metric == "fpr95" else -float("inf"),
            "val_fpr95": float("inf")}

    # Evaluate
    for src, a in candidates:
        k_logits, u_logits = get_logits(src, a)
        for T in cand_T:
            met = _eval_val_metrics_for_candidate(score_mode, float(T), k_logits, u_logits)
            cand = {"score_source": src, "T_score": float(T), "alpha": a,
                    "val_auroc": float(met["auroc"]), "val_fpr95": float(met["fpr95"])}
            if better(cand, best):
                best = cand

    return best


@torch.no_grad()
def eval_osr(
    model,
    adapter: Optional[nn.Module],
    text_feat: torch.Tensor,
    loader: DataLoader,
    device,
    thr: float,
    score_mode: str,
    score_source: str,
    T_score: float,
    mix_alpha: float,
) -> OSRResult:
    """
    Classification uses adapter logits if adapter exists, otherwise zeroshot.
    Unknown detection uses score_source (adapter/zeroshot/mix) and temperature T_score.
    Unknown if score >= thr.
    """
    all_scores = []
    all_is_unknown = []

    known_correct = 0
    known_total = 0
    osr_correct = 0
    osr_total = 0

    for images, y in loader:
        lz, la = forward_logits(model, adapter, text_feat, images, device)
        y_np = y.numpy()

        logits_ad = la.detach().cpu().numpy()
        logits_zs = lz.detach().cpu().numpy()

        # classification logits: adapter if available else zs
        cls_logits = logits_ad if adapter is not None else logits_zs
        pred = cls_logits.argmax(axis=1)

        # score logits selection
        if score_source == "adapter":
            score_logits = logits_ad
        elif score_source == "zeroshot":
            score_logits = logits_zs
        elif score_source == "mix":
            score_logits = mix_alpha * logits_ad + (1.0 - mix_alpha) * logits_zs
        else:
            raise ValueError(score_source)

        scores = score_unknown_from_logits(score_logits, T_score, score_mode)
        pred_unknown = (scores >= thr)

        is_unknown = (y_np < 0)

        # known acc
        known_mask = ~is_unknown
        if known_mask.any():
            known_total += int(known_mask.sum())
            known_correct += int((pred[known_mask] == y_np[known_mask]).sum())

        # osr acc
        osr_total += len(y_np)
        # unknown correct if predicted unknown; known correct if predicted known AND class correct
        osr_correct += int((is_unknown & pred_unknown).sum())
        osr_correct += int((known_mask & (~pred_unknown) & (pred == y_np)).sum())

        all_scores.append(scores.astype(np.float32))
        all_is_unknown.append(is_unknown.astype(np.int64))

    all_scores = np.concatenate(all_scores, 0)
    all_is_unknown = np.concatenate(all_is_unknown, 0)

    # AUROC and FPR@95
    if all_is_unknown.sum() == 0 or (all_is_unknown == 0).sum() == 0:
        auroc = 0.5
        fpr95 = 1.0
    else:
        auroc = float(roc_auc_score(all_is_unknown, all_scores))
        fpr95 = float(fpr_at_95_tpr(all_scores, all_is_unknown))

    known_acc = float(known_correct / (known_total + 1e-12))
    osr_acc = float(osr_correct / (osr_total + 1e-12))
    return OSRResult(known_acc=known_acc, osr_acc=osr_acc, auroc=auroc, fpr95=fpr95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--target_domain", type=str, required=True)
    parser.add_argument("--source_domains", type=str, default="", help="comma list; empty => all except target")
    parser.add_argument("--val_domain", type=str, default="", help="hold-out source domain for validation (must be in source_domains)")
    parser.add_argument("--unknown_classes", type=str, required=True, help="comma list of folder names to treat as unknown")

    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--baseline_zeroshot", action="store_true")

    # Adapter hyperparams
    parser.add_argument("--hidden_ratio", type=float, default=0.5, help="alias of adapter_hidden_ratio")
    parser.add_argument("--adapter_hidden_ratio", type=float, default=None)
    parser.add_argument("--max_scale", type=float, default=0.1)
    parser.add_argument("--alpha_init", type=float, default=1.0)
    parser.add_argument("--cos_reg", type=float, default=0.0)
    parser.add_argument("--early_stop", type=int, default=0, help="patience; 0 disables early stop")
    parser.add_argument("--early_stop_metric", type=str, default="auroc", choices=["known_acc", "auroc"],
                        help="metric used for early-stop / best checkpoint selection")

    # OSR scoring and prompts
    parser.add_argument("--score_mode", type=str, default="msp", choices=["msp", "energy"])
    parser.add_argument("--templates", type=str, default="", help='template ensemble split by "|". empty => "a photo of a {}"')

    # Paper-ready calibration (FPR@95-first)
    parser.add_argument("--score_from", type=str, default="adapter",
                        choices=["adapter", "zeroshot", "mix", "auto"],
                        help="which logits to use for unknown scoring; classification uses adapter if trained")
    parser.add_argument("--mix_alpha", type=float, default=0.5, help="when score_from=mix, alpha*adapter+(1-alpha)*zeroshot")
    parser.add_argument("--mix_alphas", type=str, default="0.0,0.25,0.5,0.75,1.0",
                        help="alpha grid used when score_from=auto (and for ablations)")
    parser.add_argument("--tune_temp", action="store_true", help="tune temperature on val for unknown scoring")
    parser.add_argument("--temp_candidates", type=str, default="0.2,0.3,0.4,0.5,0.6,0.8,1,1.2,1.5,2,3,4,5",
                        help="comma list of temperatures to try (for MSP/energy)")
    parser.add_argument("--calib_metric", type=str, default="fpr95", choices=["fpr95", "auroc"],
                        help="metric used to pick (score_source, alpha, T) on held-out val")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    domains = list_domains(args.data_root)
    if args.target_domain not in domains:
        raise ValueError(f"target_domain={args.target_domain} not in {domains}")

    if args.source_domains.strip():
        source_domains = [d.strip() for d in args.source_domains.split(",") if d.strip()]
    else:
        source_domains = [d for d in domains if d != args.target_domain]

    for d in source_domains:
        if d not in domains:
            raise ValueError(f"source domain '{d}' not in {domains}")

    used_domains_for_common = sorted(set([args.target_domain] + source_domains + ([args.val_domain] if args.val_domain else [])))
    common_classes = common_classes_across_domains(args.data_root, used_domains_for_common)
    if not common_classes:
        raise RuntimeError("No common classes across chosen domains. Check folder structure.")

    unknown = [c.strip() for c in args.unknown_classes.split(",") if c.strip()]
    for u in unknown:
        if u not in common_classes:
            raise ValueError(f"unknown class '{u}' not found in common classes. common={common_classes}")

    known_classes = [c for c in common_classes if c not in set(unknown)]
    known2id = {c: i for i, c in enumerate(known_classes)}

    templates = [t.strip() for t in args.templates.split("|") if t.strip()] if args.templates.strip() else ["a photo of a {}"]
    temps = [float(x.strip()) for x in args.temp_candidates.split(",") if x.strip()]
    if not temps:
        temps = [1.0]
    mix_alphas = [float(x.strip()) for x in args.mix_alphas.split(",") if x.strip() != ""]
    if not mix_alphas:
        mix_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"[Info] domains={domains}")
    print(f"[Info] source_domains={source_domains}, target_domain={args.target_domain}, val_domain={args.val_domain or '(none)'}")
    print(f"[Info] #common_classes={len(common_classes)}  #known={len(known_classes)}  #unknown={len(unknown)}")
    print(f"[Info] templates={templates}")
    print(f"[Info] score_mode={args.score_mode} score_from={args.score_from} tune_temp={args.tune_temp} calib_metric={args.calib_metric}")
    print(f"[Info] temps={temps}")
    print(f"[Info] mix_alpha={args.mix_alpha}  mix_alphas(grid)={mix_alphas}")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    text_feat = build_text_features(model, tokenizer, known_classes, device, templates)

    # ----------------------------
    # Build train/val/test datasets
    # ----------------------------
    val_domain = args.val_domain.strip()
    use_holdout_val = (val_domain != "" and val_domain in source_domains)

    train_known_samples: List[Tuple[str, int]] = []
    val_known_samples: List[Tuple[str, int]] = []
    val_unknown_samples: List[Tuple[str, int]] = []

    source_unknown_samples_for_thr: List[Tuple[str, int]] = []  # fallback unknown for val if needed

    if use_holdout_val:
        train_domains = [d for d in source_domains if d != val_domain]
        # training known
        for d in train_domains:
            sk = collect_samples(args.data_root, d, known_classes)
            train_known_samples += [(p, known2id[c]) for (p, c) in sk]
            su = collect_samples(args.data_root, d, unknown)
            source_unknown_samples_for_thr += [(p, -1) for (p, _) in su]
        # validation on held-out domain (known + unknown)
        vk = collect_samples(args.data_root, val_domain, known_classes)
        vu = collect_samples(args.data_root, val_domain, unknown)
        val_known_samples = [(p, known2id[c]) for (p, c) in vk]
        val_unknown_samples = [(p, -1) for (p, _) in vu]
        print(f"[Split] Hold-out val_domain={val_domain}: train_known={len(train_known_samples)}  val_known={len(val_known_samples)}  val_unknown={len(val_unknown_samples)}")
    else:
        # legacy split: mix all source domains then split by val_ratio
        source_known_all: List[Tuple[str, int]] = []
        source_unknown_all: List[Tuple[str, int]] = []
        for d in source_domains:
            sk = collect_samples(args.data_root, d, known_classes)
            source_known_all += [(p, known2id[c]) for (p, c) in sk]
            su = collect_samples(args.data_root, d, unknown)
            source_unknown_all += [(p, -1) for (p, _) in su]

        random.shuffle(source_known_all)
        n_val = int(len(source_known_all) * args.val_ratio)
        val_known_samples = source_known_all[:n_val]
        train_known_samples = source_known_all[n_val:]
        val_unknown_samples = source_unknown_all
        print(f"[Split] Mixed sources val_ratio={args.val_ratio}: train_known={len(train_known_samples)}  val_known={len(val_known_samples)}  val_unknown={len(val_unknown_samples)}")

    # target test set
    test_samples: List[Tuple[str, int]] = []
    tk = collect_samples(args.data_root, args.target_domain, known_classes)
    tu = collect_samples(args.data_root, args.target_domain, unknown)
    test_samples += [(p, known2id[c]) for (p, c) in tk]
    test_samples += [(p, -1) for (p, _) in tu]
    print(f"[Test] target_known={len(tk)}  target_unknown={len(tu)}  total={len(test_samples)}")

    train_loader = DataLoader(MultiDomainFolder(train_known_samples, transform=preprocess_train),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_known_loader = DataLoader(MultiDomainFolder(val_known_samples, transform=preprocess_val),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_unknown_loader = DataLoader(MultiDomainFolder(val_unknown_samples, transform=preprocess_val),
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(MultiDomainFolder(test_samples, transform=preprocess_val),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ----------------------------
    # Build / train adapter
    # ----------------------------
    adapter = None
    if not args.baseline_zeroshot:
        hidden_ratio = float(args.adapter_hidden_ratio) if args.adapter_hidden_ratio is not None else float(args.hidden_ratio)
        # Determine dim from a dummy forward
        with torch.no_grad():
            img_size = None
            if hasattr(model, "visual") and hasattr(model.visual, "image_size"):
                img_size = model.visual.image_size
            elif hasattr(model, "image_size"):
                img_size = model.image_size

            if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                h, w = int(img_size[0]), int(img_size[1])
            elif isinstance(img_size, (int, float)):
                h = w = int(img_size)
            else:
                h = w = 224

            dummy = torch.zeros(1, 3, h, w, device=device)
            feat = model.encode_image(dummy)
            dim = int(feat.shape[-1])

        adapter = Adapter(dim=dim, hidden_ratio=hidden_ratio, max_scale=args.max_scale, alpha_init=args.alpha_init).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        best_state = None
        best_metric = -1e9
        bad = 0

        def eval_val_metrics():
            # known acc on val_known
            correct = 0
            total = 0
            for images, y in val_known_loader:
                lz, la = forward_logits(model, adapter, text_feat, images, device)
                pred = la.argmax(dim=1).detach().cpu().numpy()
                y_np = y.numpy()
                correct += int((pred == y_np).sum())
                total += len(y_np)
            known_acc = float(correct / (total + 1e-12))

            # AUROC requires unknown set
            auroc = None
            if len(val_unknown_loader.dataset) > 0:
                kz, ka, _ = collect_logits_on_loader(model, adapter, text_feat, val_known_loader, device)
                uz, ua, _ = collect_logits_on_loader(model, adapter, text_feat, val_unknown_loader, device)
                # monitor with adapter logits, T=1
                met = _eval_val_metrics_for_candidate(args.score_mode, 1.0, ka, ua)
                auroc = float(met["auroc"])
            return known_acc, auroc

        for ep in range(args.epochs):
            adapter.train()
            pbar = tqdm(train_loader, desc=f"Train ep {ep+1}/{args.epochs}", ncols=100)
            for images, y in pbar:
                images = images.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.no_grad():
                    feat = model.encode_image(images)
                    feat = feat / feat.norm(dim=-1, keepdim=True)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    feat_ad = adapter(feat)
                    feat_ad = feat_ad / feat_ad.norm(dim=-1, keepdim=True)
                    if hasattr(model, "logit_scale"):
                        logit_scale = model.logit_scale.exp()
                    else:
                        logit_scale = torch.tensor(1.0, device=device)
                    logits = (feat_ad @ text_feat.T) * logit_scale
                    loss = F.cross_entropy(logits, y)

                    if args.cos_reg > 0:
                        cos = F.cosine_similarity(feat_ad, feat, dim=1)
                        loss = loss + float(args.cos_reg) * (1.0 - cos).mean()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                pbar.set_postfix(loss=float(loss.detach().cpu()))

            adapter.eval()
            val_known_acc, val_auroc = eval_val_metrics()
            metric_name = args.early_stop_metric
            if metric_name == "auroc":
                if val_auroc is None:
                    metric = val_known_acc
                    metric_name = "known_acc(fallback)"
                else:
                    metric = val_auroc
            else:
                metric = val_known_acc

            print(f"[Val] known_acc={val_known_acc:.4f}  auroc={(val_auroc if val_auroc is not None else float('nan')):.4f}  sel={metric_name}={metric:.4f}")

            improved = (metric > best_metric + 1e-8)
            if improved:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if args.early_stop and bad >= args.early_stop:
                print(f"[EarlyStop] stop at epoch {ep+1}, best={best_metric:.4f}")
                break

        if best_state is not None:
            adapter.load_state_dict(best_state)
        adapter.eval()
    else:
        adapter = None

    # ----------------------------
    # Select scoring source/alpha/temperature on val
    # ----------------------------
    if use_holdout_val and len(val_unknown_samples) == 0:
        print("[Warn] val_domain has no unknown samples; using source unknown samples for score calibration.")
        if len(source_unknown_samples_for_thr) > 0:
            val_unknown_loader = DataLoader(MultiDomainFolder(source_unknown_samples_for_thr, transform=preprocess_val),
                                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    kz, ka, _ = collect_logits_on_loader(model, adapter, text_feat, val_known_loader, device)
    if len(val_unknown_loader.dataset) > 0:
        uz, ua, _ = collect_logits_on_loader(model, adapter, text_feat, val_unknown_loader, device)
    else:
        uz = np.zeros((0, len(known_classes)), dtype=np.float32)
        ua = np.zeros((0, len(known_classes)), dtype=np.float32)

    sel = choose_score_source_and_temp(
        score_from=args.score_from,
        score_mode=args.score_mode,
        temps=temps,
        val_known_logits_zs=kz,
        val_known_logits_ad=ka,
        val_unknown_logits_zs=uz,
        val_unknown_logits_ad=ua,
        tune_temp=args.tune_temp,
        mix_alpha=args.mix_alpha,
        mix_alphas=mix_alphas,
        calib_metric=args.calib_metric,
    )
    chosen_source = sel["score_source"]
    T_score = sel["T_score"]
    chosen_alpha = sel["alpha"]

    # Build val logits for threshold selection using chosen alpha if mix
    if chosen_source == "zeroshot":
        val_known_logits = kz
        val_unknown_logits = uz
        alpha_used = None
    elif chosen_source == "adapter":
        val_known_logits = ka
        val_unknown_logits = ua
        alpha_used = None
    else:
        a = float(chosen_alpha if chosen_alpha is not None else args.mix_alpha)
        val_known_logits = a * ka + (1.0 - a) * kz
        val_unknown_logits = a * ua + (1.0 - a) * uz
        alpha_used = a

    val_known_scores = score_unknown_from_logits(val_known_logits, T_score, args.score_mode)
    val_unknown_scores = score_unknown_from_logits(val_unknown_logits, T_score, args.score_mode) if val_unknown_logits.size else np.array([], dtype=np.float32)

    if val_unknown_scores.size > 0:
        thr, f1 = pick_threshold_maxf1(val_known_scores, val_unknown_scores)
        y = np.concatenate([np.zeros_like(val_known_scores, np.int64), np.ones_like(val_unknown_scores, np.int64)], 0)
        s = np.concatenate([val_known_scores, val_unknown_scores], 0)
        val_auc = float(roc_auc_score(y, s))
        val_fpr = float(fpr_at_95_tpr(s, y))
        print(f"[Calib] score_source={chosen_source}  T_score={T_score}  alpha={(alpha_used if alpha_used is not None else 'NA')}  thr={thr:.6f}  maxF1={f1:.4f}  val_AUROC={val_auc:.4f}  val_FPR95={val_fpr:.4f}")
    else:
        thr = float(np.max(val_known_scores) + 1e-6)
        print(f"[Calib] score_source={chosen_source}  T_score={T_score}  alpha={(alpha_used if alpha_used is not None else 'NA')}  thr={thr:.6f}  no-unknown")

    # ----------------------------
    # Evaluate on target
    # ----------------------------
    print("[Run] Evaluate on target domain...")
    res = eval_osr(
        model=model,
        adapter=adapter,
        text_feat=text_feat,
        loader=test_loader,
        device=device,
        thr=thr,
        score_mode=args.score_mode,
        score_source=chosen_source,
        T_score=T_score,
        mix_alpha=(alpha_used if alpha_used is not None else args.mix_alpha),
    )

    print("========== RESULT ==========")
    print(f"Target domain: {args.target_domain}")
    print(f"Known Acc: {res.known_acc:.4f}")
    print(f"OSR Acc:  {res.osr_acc:.4f}")
    print(f"AUROC:    {res.auroc:.4f}")
    print(f"FPR@95:   {res.fpr95:.4f}")
    print("============================")


if __name__ == "__main__":
    main()
