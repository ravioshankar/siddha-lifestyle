# Kaggle GPU image training (Phase 6)

Train a **2D middle-slice CNN** on Kaggle. Local sklearn baseline: `train_image_baseline.py`.

## Recommended approach

1. Prefer **fluid-sensitive sagittal** series (see `src/dicom.select_preferred_series`).
2. Take middle slice → resize 224 → ImageNet-normalized RGB (repeat channel).
3. Backbone: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=12)`.
4. Loss: BCEWithLogits with **mask** for NaN labels; optional `pos_weight` for rare classes.
5. Split by **StudyInstanceUID** (never leak series across folds).
6. Save OOF predictions for Phase 7 fusion.

```python
# Pseudocode for Kaggle notebook
import timm, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class KneeSliceDataset(Dataset):
    def __getitem__(self, i):
        # load preferred series middle slice, preprocess
        # y: float tensor shape (12,) with NaNs → mask
        return image_chw, y, mask

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=12)
# train with masked BCE:
# loss = (bce(logits, y) * mask).sum() / mask.sum().clamp_min(1)

# Export OOF to /kaggle/working/image_oof_preds.npy
# Export weights to /kaggle/working/image_effb0.pt
```

## Local smoke test

```bash
uv run python scripts/make_sample_dicoms.py
uv run python train_image_baseline.py --cv 5
```

Record CV in `notes/score_log.md`.
