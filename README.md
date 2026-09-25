# Compact Federated Skin-Lesion Classifier

This project implements a privacy-preserving, non-IID federated learning workflow, redesigned for a laptop NVIDIA GTX 1650 (usually 4 GB VRAM). It uses a pretrained MobileNetV3-Large instead of ViT-B/16: 5.5 M parameters rather than about 86 M. A lightweight CBAM feature-attention head brings lesion-focused local and spatial attention without the memory cost of a Swin-plus-DenseNet hybrid. Client updates are trained one at a time, so GPU memory does not grow with the number of simulated hospitals.

## Important clinical note

This is a research tool, not a medical device. Do not use its output to diagnose, triage, or treat patients. Accuracy can only be established on a patient-level held-out test set that matches the intended clinical population; no implementation can honestly guarantee it exceeds every published result.

## What is improved over the reference paper

- Patient-level train/validation/test separation prevents image/patient leakage.
- Deterministic Dirichlet non-IID client splitting, class-balanced focal loss, pretrained transfer learning, and early stopping improve robust performance rather than training accuracy alone.
- FedProx reduces client drift. Server aggregation considers sample count and validation quality.
- Macro F1, balanced accuracy, per-class sensitivity/specificity, ROC-AUC, and a confusion matrix are reported, not only accuracy.
- Mixed precision, 160 px inputs, bounded GPU memory, sequential client simulation, adaptive learning-rate reduction, and one best checkpoint keep it GTX-1650 friendly.
- The optional CBAM head adapts the attention idea from multi-task lesion research to a classification-only dataset. A compact LayerNorm/dropout classifier head regularizes the CBAM-refined features without failing on small hospital batches. It does **not** claim segmentation, because segmentation masks are not part of this project's manifest.
- `tools/audit_manifest.py` identifies exact visual duplicates before combining datasets or splitting by patient, addressing the duplicate-image risk highlighted in transformer dataset research.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install the CUDA build of PyTorch suitable for your NVIDIA driver from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.
3. Prepare a CSV manifest with `image_path,label,patient_id` columns. Paths may be absolute or relative to the manifest. `patient_id` is strongly recommended.
4. For HAM10000, use the helper below. For ISIC 2019, convert its one-hot ground-truth CSV to the same three-column manifest.

```powershell
python tools/prepare_ham10000.py --metadata C:\data\HAM10000_metadata.csv --images C:\data\HAM10000_images --output data\ham10000_manifest.csv
python tools/audit_manifest.py --manifest data\ham10000_manifest.csv --output runs\ham10000_duplicate_audit.csv
python train.py --manifest data\ham10000_manifest.csv --config config\gtx1650.yaml
```

The audit exits with status `1` when it finds duplicates. Review the CSV before training: remove accidental cross-dataset copies, resolve conflicting labels, and keep every duplicate group within one patient-level split. Near-duplicates are not automatically removed because doing so needs a clinical review.

The first run downloads ImageNet weights. Training artifacts are kept in `runs/<name>/`: `best_model.pt`, `metrics.json`, `confusion_matrix.csv`, `hospital_metrics.json`, per-hospital confusion matrices in `hospital_metrics/`, and `class_names.json`. No raw images or per-client weights are copied there.

Each simulated hospital receives non-IID train, validation, and held-out test subsets. `hospital_metrics.json` reports the final federated model's full test metrics for every hospital, the combined test metrics, mean/standard deviation across hospitals, and worst-hospital macro F1. `metrics.json` also records local validation metrics for each selected hospital in every training round.

Every metric report includes accuracy, balanced accuracy, micro/macro/weighted precision-recall-F1, one-vs-rest ROC-AUC, average precision, top-2 accuracy, Cohen's kappa, Matthews correlation coefficient, and log loss. For every class it includes precision (PPV), sensitivity/recall, specificity, negative predictive value, F1, ROC-AUC, average precision, false-positive/false-negative rates, TP/TN/FP/FN counts, and support. Confusion matrices are saved separately.

## Verify before using medical data

Run `python tools/check_setup.py`. It must say `CUDA available: True` and identify the GTX 1650. Then run `python tools/smoke_test.py`. This creates temporary artificial images, performs one federated training round, checks evaluation and aggregation, prints `SMOKE TEST PASSED`, and leaves no data or model files behind.

## Tune safely

- If CUDA reports out-of-memory, set `batch_size: 8` and `image_size: 144`.
- Start with `rounds: 8` to verify data. The default 30 rounds is a realistic fine-tuning run, not a claim of clinical readiness.
- Use `clients: 3` to reproduce the paper's simulation. Federated simulation is not a substitute for real institutional deployment; raw data still stays in each configured local client in a real deployment.
- Set `use_cbam: false` if you need the original MobileNetV3 head for an ablation. The default `true` setting is still suitable for 4 GB VRAM at the supplied 160 px input size.
- Keep `strong_augmentation: false` for the baseline. Enable it only as a documented ablation: it adds low-probability blur, perspective, and erasing transforms, which may improve robustness but can also affect probability calibration.
- The global validation macro-F1 controls the learning-rate schedule. After `lr_patience` rounds without improvement, each client starts future rounds at `lr_factor` times the previous rate, down to `min_learning_rate`.
