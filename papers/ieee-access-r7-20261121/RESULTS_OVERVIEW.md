# Results and data for the IEEE Access R7 revision

Numerical Stability and Threshold Transfer in Validation-Tuned Open-Set Detection

Updated 21 November 2026. These are exports of the checked manuscript evidence, not new experiments or a new selection of favorable outcomes. Displayed decimals match the paper; CSV files preserve the underlying numeric values.

## Primary disjoint-unknown panels

Equal-unit means: PACS 84 units and CIFAR-10 15 units. This table mirrors manuscript Table 2. The panels use disjoint validation-unknown and target-unknown classes. FPR@95 is a labeled-target ROC metric and is separate from frozen-threshold recall.

| Method | PACS AUROC | PACS FPR@95 | CIFAR-10 AUROC | CIFAR-10 FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| MSP | 0.948 | 0.152 | 0.842 | 0.443 |
| Temp-MSP, FPR objective | 0.951 | 0.147 | 0.870 | 0.387 |
| autoMixTemp | 0.952 | 0.145 | 0.870 | 0.384 |
| Temp-MSP, AUROC objective | 0.952 | 0.144 | 0.871 | 0.385 |
| Energy, T=1 | 0.942 | 0.194 | 0.924 | 0.315 |
| Energy, tuned T | 0.946 | 0.187 | 0.922 | 0.321 |
| MCM, canonical prompt | 0.945 | 0.162 | 0.880 | 0.384 |
| MCM, matched prompts | 0.946 | 0.163 | 0.877 | 0.370 |
| Maximum cosine | 0.943 | 0.192 | 0.927 | 0.309 |
| NegLabel port, standard | 0.925 | 0.224 | 0.919 | 0.276 |
| NegLabel port, name-excluded | 0.925 | 0.225 | 0.919 | 0.276 |

[Download all primary means](data/main_table2_primary_means.csv). Full unit-level evidence, probability diagnostics and threshold scans are retained in the complete evidence archive.

## Separate Office-Home panel

All 60 units, seven prespecified scores and three optimization seeds are retained. This table mirrors manuscript Table 14. The smaller sampled budget differs from the primary panels; dataset panels are not pooled and seeds reuse image manifests.

| Method | AUROC | FPR@95 | Target unknown recall | Known rejection |
| --- | ---: | ---: | ---: | ---: |
| MSP | 0.838 | 0.465 | 0.932 | 0.447 |
| Temp-MSP | 0.858 | 0.431 | 0.922 | 0.403 |
| Energy, T=1 | 0.791 | 0.665 | 0.866 | 0.622 |
| Energy, tuned T | 0.809 | 0.633 | 0.875 | 0.590 |
| Mix, alpha=0.5 | 0.858 | 0.433 | 0.920 | 0.402 |
| Zero-shot, tuned T | 0.857 | 0.435 | 0.920 | 0.401 |
| autoMixTemp | 0.859 | 0.430 | 0.918 | 0.395 |

[Summary CSV](data/main_table14_officehome_means.csv) | [All 420 score-unit records](data/officehome_all60_all7_scores.csv).

## Completed controls

- [Detector-specific checkpoint/selector control: all 264 records](data/checkpoint_detector_selector_units.csv). The control covers 33 complete training trajectories; this is not independent detector-family retraining.
- [Official LoCoOp: all 264 score-fit records](data/locoop_both_lambdas_all66_fits_all4_scores.csv). There are 66 prompt fits on 33 distinct matched seed-0 roles, at both fixed regularization weights 0.1 and 0.25. Official and float64 MCM/GL-MCM realizations are all retained.
- [All matched LoCoOp summary realizations](data/matched_locoop_all_realizations_summary.csv). The prompt-training budget is 50 epochs, versus the adapter's maximum 20 epochs. Training objectives and budgets differ; target outcomes select neither checkpoint nor regularization weight.

## Interpretation and scope

Stable log-odds preserve distinctions lost by subtractive float32 MSP. Improved detection ranking does not imply improved probability reliability or guaranteed 95-percent target recall after threshold transfer. Some baselines outperform autoMixTemp, and the data do not establish a consistent interior-mixing advantage over Temp-MSP. Shared images, class roles and training/validation data limit inference to the reported conditional fitted-panel evidence.

## Files and verification

[Supplementary analyses PDF](downloads/supplementary_analyses.pdf) | [Complete evidence/code ZIP](downloads/supplementary_evidence.zip) | [File hashes](SHA256SUMS.txt)

The archive contains processed numerical evidence and analysis code, not raw image datasets, pretrained models, checkpoints or raw logits. Extract the archive and run `python -B -s scripts/verify_final_package.py` from its root for the portable numerical consistency check. See [the evidence index](README.md) for details.
