# Updated evidence for the IEEE Access revision

**Numerical Stability and Threshold Transfer in Validation-Tuned Open-Set Detection**

Updated 21 November 2026. The checked R7 numerical evidence is now organized separately from the historical training code below.

- **[Open the updated results and data tables](papers/ieee-access-r7-20261121/RESULTS_OVERVIEW.md)**
- [R7 evidence index and verification instructions](papers/ieee-access-r7-20261121/README.md)
- [Supplementary analyses PDF](papers/ieee-access-r7-20261121/downloads/supplementary_analyses.pdf)
- [Download the full numerical evidence and code ZIP](https://raw.githubusercontent.com/lerc487/clip-adapter-osdg/main/papers/ieee-access-r7-20261121/downloads/supplementary_evidence.zip)

The primary panels comprise 84 PACS and 15 CIFAR-10 units with disjoint validation/target unknown classes. The separate Office-Home panel includes all 60 units, and the matched official LoCoOp control retains both regularization weights and all 66 prompt fits on 33 distinct sample-role units. These are finite-panel results, not a transferable 95-percent recall guarantee or evidence of universal autoMixTemp superiority. All variants and adverse results are retained.

**Version note:** The dated R7 directory is the entry point for this revised manuscript. The original `src/`, `scripts/` and `docs/` files below are retained as historical project code, not substituted for the revised evidence or its documented dependencies. The original README is also preserved [verbatim](papers/ieee-access-r7-20261121/README_before_update.md). No raw datasets, checkpoints or raw logits are redistributed.

---

## Historical project documentation

# Domain-Held-Out Validation & Temperature Calibration for CLIP Adapters (OSDG)

This repository contains code for the manuscript:

**"Domain-Held-Out Validation and Temperature Calibration: Stabilizing Rejection Performance of CLIP Adapters for Open-Set Domain Generalization"**.

The code supports open-set domain generalization experiments using CLIP adapters and a folder-based dataset format:
`DATA_ROOT/domain/class/*.jpg`.

---

## Environment

We recommend using a conda environment (Windows examples):

```bash
conda create -n icip_osr python=3.10 -y
conda activate icip_osr

# core dependencies
pip install -U torch open_clip_torch scikit-learn numpy tqdm pillow pyyaml

# for public dataset download/export
pip install -U datasets
```

> If you encounter CUDA/torch issues, install the PyTorch build that matches your CUDA version from the official PyTorch website.

---

## Data preparation (PACS)

We use the publicly available **PACS** dataset and access it via the **Hugging Face `datasets`** library.  
We **do not redistribute** any dataset files. We provide a script to download and export PACS into the folder structure required by this repo:

`DATA_ROOT/domain/class/*.jpg`

### Export PACS to folders

```bash
python scripts/export_pacs_to_folders.py --out_dir C:\icip_osr\data\PACS
```

After export, the expected domains are:

- `art_painting`
- `cartoon`
- `photo`
- `sketch`

and classes (for the HF version used in our experiments) include:

- `dog`, `elephant`, `giraffe`, `guitar`, `horse`, `house`, `person`

---

## Run (minimal)

Minimal runner (recommended to start with):

```bash
python src/paper_run_minimal.py --data_root C:\icip_osr\data\PACS --num_splits 10 --seeds 0,1 --epochs 20 --early_stop 3
```

Additional/extended runners and analysis scripts are provided in `src/` and example commands can be found in:

- `docs/runcode.txt`
- `docs/README_FPR95.txt`

---

## Notes

- This repo expects data organized as `DATA_ROOT/domain/class/*.jpg`.
- The PACS export script downloads from public sources via Hugging Face Datasets.
- Please do not upload dataset files to the repository.

---

## Citation

If you use this code, please cite the paper (to be updated upon publication).

---

## License

Add a license file (e.g., MIT) if you plan to redistribute or reuse the code openly.
