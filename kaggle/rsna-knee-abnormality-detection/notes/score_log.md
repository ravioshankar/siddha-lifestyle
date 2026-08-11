# Experiment / score log

Track CV and leaderboard scores after each phase. Update after every submission.

| Phase | Model | Local CV mean AUC | CV std | Public LB | Notes | Date |
|-------|-------|-------------------|--------|-----------|-------|------|
| 3 | TF-IDF + OvR LogReg (text, expert-labeled only) | 0.9707 | 0.0310 | — | Synthetic sample CSVs; NaN→0 in train | 2026-08-07 |
| 4 | Same → `submission_text_baseline.csv` | 0.9707 | | _pending_ | Needs `~/.kaggle/access_token` + Notebook submit; test has no Report | 2026-08-07 |
| 5 | Masked per-label LR + weak labels (prec≥0.7) | 0.9829 | 0.0071 | — | Beat Phase 3 by +0.0122; see `notes/weak_label_qc.md` | 2026-08-07 |
| 6 | 2D middle-slice image baseline (sklearn) | 1.0000 | 0.0000 | — | Synthetic blob DICOMs — re-run on real MRI | 2026-08-07 |
| 7 | Late fusion average (text+image OOF) | 0.9995 | — | _pending_ | Overlap n=84; image-dominant on empty-report test | 2026-08-07 |

## How to update

```bash
uv run python train_text_baseline.py --cv 5
uv run python train_text_improved.py --cv 5
uv run python train_image_baseline.py --cv 5
uv run python fuse_predictions.py
# On Kaggle: follow notebooks/02_kaggle_submit.md then record Public LB above
```

## CV vs LB gap

- Large drop with text-only usually means **test reports are missing/empty**.
- Prefer `submission_fusion.csv` (image-dominant when reports empty).
- Synthetic local AUCs are optimistic; replace data via `uv run bash scripts/download_csvs.sh` once credentials exist.
