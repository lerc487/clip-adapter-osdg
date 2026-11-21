# Numerical Stability and Threshold Transfer in Validation-Tuned Open-Set Detection

## Revised manuscript numerical evidence

**[Open the results tables and browse updated data](RESULTS_OVERVIEW.md).**

Prepared on 21 November 2026 from the checked R7 evidence snapshot of 20 November 2026. No experiments, outcomes or analysis choices were changed for this repository update. This landing page identifies the completed panels and provides directly browsable per-unit data.

### Downloads

- [Supplementary analyses PDF](downloads/supplementary_analyses.pdf): Tables S1-S19 and Figure S1.
- [Complete numerical evidence and analysis code](downloads/supplementary_evidence.zip): the existing checked archive, preserved byte for byte.
- [SHA-256 file hashes](SHA256SUMS.txt).

### Browse the updated data

| Evidence | Browse CSV | Records |
| --- | --- | ---: |
| Complete detector-specific checkpoint/selector control | [Per-unit records](data/checkpoint_detector_selector_units.csv) | 264 |
| Office-Home, all 60 units and all seven scores | [Per-unit records](data/officehome_all60_all7_scores.csv) | 420 |
| Official LoCoOp, both fixed regularization weights and all four score realizations | [Per-fit records](data/locoop_both_lambdas_all66_fits_all4_scores.csv) | 264 |

Additional indexes:

- [Consolidated manuscript numerical evidence](data/E7_final_numeric_evidence.json).
- [Matched LoCoOp comparison summary](data/E8_locoop_matched_summary.json).

The primary disjoint-unknown panels comprise 84 PACS and 15 CIFAR-10 units. A separate matched RN50 control has 33 units. Office-Home has 60 units across three optimization seeds and uses a separate smaller sampled budget; its panels are not pooled with the primary datasets. LoCoOp comprises 66 prompt fits on 33 distinct matched seed-0 roles at two fixed regularization weights, not 66 independent sample-role units. All outcome variants are retained, including unfavorable probability, threshold-transfer and baseline comparisons. Class roles and images are reused across some configurations and seeds.

### Verify the downloaded evidence

Extract `downloads/supplementary_evidence.zip`, open a terminal at the extracted root, and run:

```text
python -B -s scripts/verify_final_package.py
```

This portable consistency check does not load raw images, model states or logits. The archive provides numerical evidence and code, not a standalone end-to-end neural-network reproduction bundle. Large raw datasets, pretrained models, checkpoints and raw logits remain separate dependencies. Training and queue scripts are archival code and should not be run merely to verify the numerical archive.

### Version handling

This update supplements the manuscript's earlier records. Earlier releases and data should remain available as historical versions. The named full archive and the extracted browse files above match the checked R7 artifacts; no earlier release should be silently relabeled as this version.
