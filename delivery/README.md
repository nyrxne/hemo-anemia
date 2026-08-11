# H.E.M.A. — Baseline Model Package

## Why this wasn't fully trained here

This working environment's network access is restricted to a small allowlist
(PyPI, GitHub, npm, etc.) and does **not** include `download.pytorch.org`, which
is where PyTorch/torchvision serve ImageNet-pretrained weights. A direct test
confirmed this (HTTP 403). Since your materials list already specifies
Google Colab, the real baseline should be trained there, where pretrained
weights download normally and a GPU is available.

## What's in this folder

- **`HEMA_baseline_training.ipynb`** — the notebook to run in Colab. Upload
  `CP-AnemiC_split_ready.zip` (from the cleaning/split step) when prompted, then
  Run All. Produces `results.json`, `model_weights.pt`, and a comparison plot
  (`in_domain_vs_unseen_site.png`) ready to use as one of the Expo's required
  images.
- **`dataset.py`, `model.py`, `train.py`** — the same pipeline as a plain Python
  script (for running outside a notebook, e.g. `python train.py --data_root ...`).
- **`smoke_test_results_DO_NOT_USE_AS_FINAL/`** — a 2-epoch, random-init run
  executed in this sandbox purely to confirm the code, data loading, and metric
  computation work correctly end-to-end. AUROC ≈ 0.49 (chance) is expected and
  correct for this run — it has no pretrained features and almost no training.
  **These numbers are not a real result and must not appear in the submission.**

## Design decisions worth stating in your writeup

- **Architecture:** EfficientNet-B0, ImageNet-pretrained, classifier head replaced
  for binary output. `mobilenet_v2` is also implemented as a lighter alternative.
- **Preprocessing:** images are RGBA crops of the conjunctiva with a transparent
  background and a highly elongated aspect ratio. Each image is composited onto a
  solid black background (transparent area carries no clinical signal) and then
  resized+letterboxed to a square, preserving aspect ratio rather than squashing
  the crescent shape.
- **Augmentation:** kept deliberately mild — horizontal flip, small rotation,
  light brightness/contrast jitter. Aggressive color jitter was avoided on purpose,
  since color (pallor vs. redness) is the actual clinical signal being learned.
- **Training:** two-phase fine-tuning — head-only first (backbone frozen), then
  full-network fine-tuning at a lower learning rate. Standard practice for a small
  (~250 image) training set to avoid destroying pretrained features too early.
- **Class imbalance:** handled with inverse-frequency class weights computed from
  the training set only.
- **Model selection:** best checkpoint chosen by validation AUROC — the
  unseen-site test set is touched exactly once, at the very end, for reporting only.
- **Metrics reported:** accuracy, sensitivity (Anemic recall), specificity
  (Non-anemic recall), AUROC, and Brier score (calibration) — not accuracy alone.

## Next step

Run the notebook in Colab, then bring the `results.json` back here (or paste the
numbers) so we can move into Day 4: the core in-domain-vs-unseen-site comparison
and the distribution-shift diagnosis using the `HOSPITAL`/`REGION`/`GENDER`/
`Age(Months)` fields already in `split_assignment.csv`.
